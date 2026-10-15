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
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
enum class Scenario { SlowUpload, DrainedFailure, HealthyStop };

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

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
    while (size != 0) {
        pollfd ready{fd, POLLIN, 0};
        require(::poll(&ready, 1, 2000) > 0, "test handshake timed out");
        const auto received = ::read(fd, bytes, size);
        if (received < 0 && errno == EINTR) { continue; }
        require(received > 0, "test child closed the handshake");
        bytes += received;
        size -= static_cast<std::size_t>(received);
    }
}

int child_server(Scenario scenario, int ready_fd, int command_fd) {
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
    ninfer::serve::EngineFailureMonitor monitor(1s);
    std::thread listener([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    monitor.start([&] { return available.load(); }, [&] {
        server.stop();
        const int stopped = 2;
        write_bytes(ready_fd, &stopped, sizeof(stopped));
    });
    write_bytes(ready_fd, &port, sizeof(port));
    char command{};
    read_bytes(command_fd, &command, sizeof(command));
    if (scenario == Scenario::HealthyStop) {
        server.stop();
    } else {
        available.store(false);
    }
    // The slow-upload case must be terminated by the production monitor while this join blocks.
    listener.join();
    monitor.stop();
    require(monitor.failed() == (scenario != Scenario::HealthyStop),
            "shutdown reported the wrong engine failure state");
    if (scenario == Scenario::HealthyStop) {
        // Staying alive beyond the grace verifies that a normal stop did not arm a fatal timer.
        std::this_thread::sleep_for(1200ms);
    }
    return 0;
}

int connect_client(int port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create test HTTP client");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to connect test HTTP client");
    }
    return fd;
}

std::string read_response(int fd, bool headers_only) {
    std::string response;
    const auto deadline = Clock::now() + 2s;
    for (;;) {
        pollfd ready{fd, POLLIN, 0};
        require(Clock::now() < deadline && ::poll(&ready, 1, 100) >= 0,
                "test HTTP response timed out");
        if (ready.revents == 0) { continue; }
        char buffer[1024];
        const auto size = ::recv(fd, buffer, sizeof(buffer), 0);
        require(size >= 0, "failed to receive test HTTP response");
        if (size == 0) { return response; }
        response.append(buffer, static_cast<std::size_t>(size));
        if (headers_only && response.find("\r\n\r\n") != std::string::npos) { return response; }
    }
}

void run_case(Scenario scenario) {
    int ready[2], commands[2];
    require(::pipe(ready) == 0 && ::pipe(commands) == 0, "failed to create test pipes");
    const pid_t child = ::fork();
    require(child >= 0, "failed to fork test HTTP server");
    if (child == 0) {
        ::close(ready[0]);
        ::close(commands[1]);
        try {
            std::_Exit(child_server(scenario, ready[1], commands[0]));
        } catch (...) {
            std::_Exit(90);
        }
    }
    ::close(ready[1]);
    ::close(commands[0]);
    int client = -1;
    bool reaped = false;
    try {
        int port{};
        read_bytes(ready[0], &port, sizeof(port));
        if (scenario != Scenario::HealthyStop) {
            client = connect_client(port);
            if (scenario == Scenario::SlowUpload) {
                const std::string request =
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                    "Content-Length: 1000000\r\nExpect: 100-continue\r\n\r\n";
                write_bytes(client, request.data(), request.size());
                require(read_response(client, true).find("100 Continue") != std::string::npos,
                        "slow upload did not reach the HTTP body reader");
                write_bytes(client, "{", 1);
            } else {
                const std::string request = "GET /error HTTP/1.1\r\nHost: localhost\r\n\r\n";
                write_bytes(client, request.data(), request.size());
                int entered{};
                read_bytes(ready[0], &entered, sizeof(entered));
                require(entered == 1, "failure response worker did not start");
            }
        }
        const char command = 's';
        write_bytes(commands[1], &command, sizeof(command));
        if (scenario != Scenario::HealthyStop) {
            int stopped{};
            read_bytes(ready[0], &stopped, sizeof(stopped));
            require(stopped == 2, "engine failure did not stop accepting HTTP connections");
        }
        const auto stopped_at = Clock::now();
        if (scenario == Scenario::DrainedFailure) {
            const auto response = read_response(client, false);
            require(response.find("500 Internal Server Error") != std::string::npos &&
                        response.find(R"({"error":"engine unavailable"})") != std::string::npos,
                    "engine failure grace did not preserve the HTTP error response");
        }
        int status{};
        int upload_bytes = 0;
        for (;;) {
            const auto result = ::waitpid(child, &status, WNOHANG);
            require(result >= 0, "failed to observe child exit");
            if (result == child) {
                reaped = true;
                break;
            }
            require(Clock::now() - stopped_at < 2s, "engine failure shutdown remained blocked");
            if (scenario == Scenario::SlowUpload && ::send(client, " ", 1, 0) == 1) {
                ++upload_bytes;
            }
            std::this_thread::sleep_for(20ms);
        }
        require(WIFEXITED(status), "test server was killed instead of exiting");
        if (scenario == Scenario::SlowUpload) {
            require(WEXITSTATUS(status) == EXIT_FAILURE, "slow upload bypassed fatal shutdown");
            require(upload_bytes > 0 && Clock::now() - stopped_at >= 800ms &&
                        Clock::now() - stopped_at < 1800ms,
                    "fatal shutdown did not honor its grace deadline");
        } else {
            require(WEXITSTATUS(status) == 0, "graceful shutdown incorrectly forced process exit");
        }
    } catch (...) {
        if (!reaped) {
            ::kill(child, SIGKILL);
            ::waitpid(child, nullptr, 0);
        }
        if (client >= 0) { ::close(client); }
        ::close(ready[0]);
        ::close(commands[1]);
        throw;
    }
    if (client >= 0) { ::close(client); }
    ::close(ready[0]);
    ::close(commands[1]);
}

} // namespace

int main() {
    ::signal(SIGPIPE, SIG_IGN);
    try {
        run_case(Scenario::SlowUpload);
        run_case(Scenario::DrainedFailure);
        run_case(Scenario::HealthyStop);
        std::cout << "ok: slow upload deadline, error response grace, healthy shutdown\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
