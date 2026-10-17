#include "serve/engine_failure_monitor.h"

#include <httplib.h>

#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using ninfer::serve::EngineFailureMonitor;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Fd {
    int value = -1;
    ~Fd() { close(); }
    void close() {
        if (value >= 0) { ::close(value); }
        value = -1;
    }
};

struct Pipe {
    Fd reader, writer;
    Pipe() {
        int fds[2];
        require(::pipe(fds) == 0, "failed to create handshake pipe");
        reader.value = fds[0];
        writer.value = fds[1];
    }
};

struct Child {
    pid_t pid;
    bool reaped = false;
    ~Child() {
        if (!reaped && pid > 0) {
            ::kill(pid, SIGKILL);
            while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }
};

struct MonitorStop {
    EngineFailureMonitor& monitor;
    ~MonitorStop() { monitor.stop(); }
};

void write_bytes(int fd, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    while (size != 0) {
        const auto written = ::write(fd, bytes, size);
        if (written < 0 && errno == EINTR) { continue; }
        require(written > 0, "failed to write test handshake");
        bytes += written;
        size -= static_cast<std::size_t>(written);
    }
}

void read_bytes(int fd, void* data, std::size_t size) {
    auto* bytes = static_cast<char*>(data);
    const auto deadline = Clock::now() + 2s;
    while (size != 0) {
        pollfd ready{fd, POLLIN, 0};
        require(Clock::now() < deadline, "test handshake timed out");
        const auto polled = ::poll(&ready, 1, 20);
        if (polled < 0 && errno == EINTR) { continue; }
        require(polled >= 0, "failed to poll test handshake");
        if (polled == 0) { continue; }
        const auto received = ::read(fd, bytes, size);
        if (received < 0 && errno == EINTR) { continue; }
        require(received > 0, "test child closed the handshake");
        bytes += received;
        size -= static_cast<std::size_t>(received);
    }
}

int wait_child(Child& child, Clock::time_point deadline) {
    int status = 0;
    for (;;) {
        const auto result = ::waitpid(child.pid, &status, WNOHANG);
        if (result < 0 && errno == EINTR) { continue; }
        require(result >= 0, "failed to observe owned child exit");
        if (result == child.pid) {
            child.reaped = true;
            return status;
        }
        require(Clock::now() < deadline, "owned child exceeded parent timeout");
        std::this_thread::sleep_for(5ms);
    }
}

void lifecycle() {
    EngineFailureMonitor monitor(5s);
    monitor.stop();
    require(!monitor.failed(), "unstarted monitor reports failure");
    std::atomic<int> stopped{0};
    for (int round = 0; round < 2; ++round) {
        std::promise<void> checked;
        auto observed = checked.get_future();
        std::atomic<bool> first{true};
        MonitorStop stop_on_error{monitor};
        monitor.start([&] {
            if (first.exchange(false)) { checked.set_value(); }
            return true;
        }, [&] { ++stopped; });
        require(observed.wait_for(2s) == std::future_status::ready,
                "healthy monitor did not check availability");
        bool duplicate_rejected = false;
        try {
            monitor.start([] { return true; }, [] {});
        } catch (const std::logic_error&) { duplicate_rejected = true; }
        require(duplicate_rejected, "monitor accepted a duplicate start");
        monitor.finish_shutdown();
        monitor.stop();
        require(!monitor.failed() && stopped == 0,
                "normal stop armed failure or called stop_accepting");
    }
    std::promise<void> failure_seen;
    auto failure = failure_seen.get_future();
    std::atomic<int> checks{0};
    MonitorStop stop_failure_on_error{monitor};
    monitor.start([&] { ++checks; return false; }, [&] {
        if (++stopped == 1) { failure_seen.set_value(); }
    });
    require(failure.wait_for(2s) == std::future_status::ready && monitor.failed(),
            "failure was not published before the stop callback");
    std::this_thread::sleep_for(150ms);
    require(stopped == 1 && checks == 1, "failure stop callback or availability check repeated");
    monitor.stop();
    require(monitor.failed(), "drained failure stop cleared failed state");
    std::promise<void> restarted;
    auto restart = restarted.get_future();
    std::atomic<bool> first{true};
    MonitorStop stop_restart_on_error{monitor};
    monitor.start([&] {
        if (first.exchange(false)) { restarted.set_value(); }
        return true;
    }, [&] { ++stopped; });
    require(restart.wait_for(2s) == std::future_status::ready && !monitor.failed(),
            "restart did not reset the previous failure state");
    monitor.stop();
    require(stopped == 1, "healthy restart stopped accepting");
}

std::int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch()).count();
}

void deadline_exit(bool production_default) {
    Pipe ready;
    const auto pid = ::fork();
    require(pid >= 0, "failed to fork deadline child");
    if (pid == 0) {
        ready.reader.close();
        try {
            const auto blocked_drain = [&](EngineFailureMonitor& monitor) {
                monitor.start([] { return false; }, [&] {
                    const auto armed = steady_ns();
                    write_bytes(ready.writer.value, &armed, sizeof(armed));
                });
                std::promise<void> never_drained;
                never_drained.get_future().wait();
            };
            if (production_default) {
                EngineFailureMonitor monitor;
                blocked_drain(monitor);
            } else {
                EngineFailureMonitor monitor(300ms);
                blocked_drain(monitor);
            }
            std::_Exit(91);
        } catch (...) { std::_Exit(90); }
    }
    Child child{pid};
    ready.writer.close();
    std::int64_t armed = 0;
    read_bytes(ready.reader.value, &armed, sizeof(armed));
    const auto status = wait_child(child, Clock::now() + (production_default ? 8s : 3s));
    const auto elapsed = std::chrono::nanoseconds(steady_ns() - armed);
    require(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_FAILURE,
            "blocked drain did not exit through production EXIT_FAILURE");
    require(elapsed >= (production_default ? 4800ms : 200ms) &&
                elapsed < (production_default ? 8s : 3s),
            "monitor did not honor the short or production-default five-second grace");
}

enum class HttpScenario { SlowUpload, DrainedFailure, HealthyStop };

int child_server(HttpScenario scenario, int ready_fd, int command_fd) {
    httplib::Server server;
    server.set_read_timeout(0, 500000);
    std::atomic<bool> available{true};
    server.Post("/v1/chat/completions", [](const auto&, auto& response) {
        response.set_content("{}", "application/json");
    });
    server.Get("/error", [&](const auto&, auto& response) {
        const int entered = 1;
        write_bytes(ready_fd, &entered, sizeof(entered));
        while (available.load()) { std::this_thread::sleep_for(1ms); }
        response.status = 500;
        response.set_content(R"({"error":"engine unavailable"})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    require(port > 0, "failed to bind test HTTP server");
    EngineFailureMonitor monitor(1s);
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    monitor.start([&] { return available.load(); }, [&] {
        server.stop();
        const int stopped = 2;
        write_bytes(ready_fd, &stopped, sizeof(stopped));
    });
    write_bytes(ready_fd, &port, sizeof(port));
    char command = 0;
    read_bytes(command_fd, &command, sizeof(command));
    if (scenario == HttpScenario::HealthyStop) {
        server.stop();
    } else {
        available.store(false);
    }
    listener.join();
    monitor.finish_shutdown();
    if (scenario != HttpScenario::HealthyStop) {
        // A blocked destructor/log sink after HTTP drain must never be reached on fatal.
        std::promise<void> blocked_cleanup;
        blocked_cleanup.get_future().wait();
    }
    require(!monitor.failed(), "healthy HTTP drain reported engine failure");
    if (scenario == HttpScenario::HealthyStop) { std::this_thread::sleep_for(1200ms); }
    return 0;
}

int connect_client(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create local HTTP client");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to connect local HTTP client");
    }
    return fd;
}

std::string read_response(int fd, bool headers_only) {
    std::string response;
    const auto deadline = Clock::now() + 2s;
    for (;;) {
        pollfd ready{fd, POLLIN, 0};
        require(Clock::now() < deadline, "test HTTP response timed out");
        const auto polled = ::poll(&ready, 1, 20);
        if (polled < 0 && errno == EINTR) { continue; }
        require(polled >= 0, "failed to poll local HTTP response");
        if (polled == 0) { continue; }
        char buffer[1024];
        const auto size = ::recv(fd, buffer, sizeof(buffer), 0);
        if (size < 0 && errno == EINTR) { continue; }
        require(size >= 0, "failed to receive local HTTP response");
        if (size == 0) { return response; }
        response.append(buffer, static_cast<std::size_t>(size));
        if (headers_only && response.find("\r\n\r\n") != std::string::npos) { return response; }
    }
}

void http_drain(HttpScenario scenario) {
    Pipe ready, commands;
    const auto pid = ::fork();
    require(pid >= 0, "failed to fork local HTTP server");
    if (pid == 0) {
        ready.reader.close();
        commands.writer.close();
        try {
            std::_Exit(child_server(scenario, ready.writer.value, commands.reader.value));
        } catch (...) { std::_Exit(90); }
    }
    Child child{pid};
    ready.writer.close();
    commands.reader.close();
    int port = 0;
    read_bytes(ready.reader.value, &port, sizeof(port));
    Fd client;
    if (scenario != HttpScenario::HealthyStop) {
        client.value = connect_client(port);
        const std::string request = scenario == HttpScenario::SlowUpload
            ? "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
              "Content-Length: 1000000\r\nExpect: 100-continue\r\n\r\n"
            : "GET /error HTTP/1.1\r\nHost: localhost\r\n\r\n";
        write_bytes(client.value, request.data(), request.size());
        if (scenario == HttpScenario::SlowUpload) {
            require(read_response(client.value, true).find("100 Continue") != std::string::npos,
                    "slow upload did not reach the HTTP body reader");
            write_bytes(client.value, "{", 1);
        } else {
            int entered = 0;
            read_bytes(ready.reader.value, &entered, sizeof(entered));
            require(entered == 1, "failure response worker did not enter");
        }
    }
    const char command = 's';
    write_bytes(commands.writer.value, &command, sizeof(command));
    if (scenario != HttpScenario::HealthyStop) {
        int stopped = 0;
        read_bytes(ready.reader.value, &stopped, sizeof(stopped));
        require(stopped == 2, "engine failure did not stop HTTP acceptance");
    }
    const auto stopped_at = Clock::now();
    if (scenario == HttpScenario::DrainedFailure) {
        const auto response = read_response(client.value, false);
        require(response.find("500 Internal Server Error") != std::string::npos &&
                    response.find(R"({"error":"engine unavailable"})") != std::string::npos,
                "failure grace did not preserve the complete error response");
    }
    int status = 0;
    int upload_bytes = 0;
    for (;;) {
        const auto result = ::waitpid(pid, &status, WNOHANG);
        if (result < 0 && errno == EINTR) { continue; }
        require(result >= 0, "failed to observe HTTP child exit");
        if (result == pid) { child.reaped = true; break; }
        require(Clock::now() - stopped_at < 3s, "local HTTP child exceeded parent timeout");
        if (scenario == HttpScenario::SlowUpload && ::send(client.value, " ", 1, 0) == 1) {
            ++upload_bytes;
        }
        std::this_thread::sleep_for(20ms);
    }
    require(WIFEXITED(status), "HTTP child was signalled instead of exiting");
    if (scenario == HttpScenario::SlowUpload) {
        require(WEXITSTATUS(status) == EXIT_FAILURE && upload_bytes > 0 &&
                    Clock::now() - stopped_at >= 800ms,
                "active slow upload bypassed the monitor failure deadline");
    } else if (scenario == HttpScenario::DrainedFailure) {
        require(WEXITSTATUS(status) == EXIT_FAILURE,
                "fatal HTTP drain entered cleanup instead of exiting");
    } else {
        require(WEXITSTATUS(status) == 0, "graceful HTTP drain forced process exit");
    }
}

} // namespace

int main() {
    ::signal(SIGPIPE, SIG_IGN);
    try {
        lifecycle();
        deadline_exit(false);
        deadline_exit(true);
        http_drain(HttpScenario::SlowUpload);
        http_drain(HttpScenario::DrainedFailure);
        http_drain(HttpScenario::HealthyStop);
        std::cout << "ok: monitor lifecycle, bounded exit and local HTTP drain\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
