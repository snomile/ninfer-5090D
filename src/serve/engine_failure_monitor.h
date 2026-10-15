#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace ninfer::serve {

// Only engine failure arms the deadline. Ordinary signal-driven HTTP shutdown remains graceful.
class EngineFailureMonitor {
public:
    explicit EngineFailureMonitor(
        std::chrono::milliseconds shutdown_grace = std::chrono::seconds(5))
        : shutdown_grace_(shutdown_grace) {}

    ~EngineFailureMonitor() { stop(); }

    void start(std::function<bool()> is_available, std::function<void()> stop_accepting) {
        if (thread_.joinable()) { throw std::logic_error("engine monitor is already running"); }
        {
            std::lock_guard lock(mutex_);
            stopping_ = false;
        }
        failed_.store(false, std::memory_order_release);
        thread_ = std::thread([this, is_available = std::move(is_available),
                               stop_accepting = std::move(stop_accepting)] {
            std::unique_lock lock(mutex_);
            for (;;) {
                if (stopping_) { return; }
                lock.unlock();
                const bool available = is_available();
                lock.lock();
                if (stopping_) { return; }
                if (!available) {
                    failed_.store(true, std::memory_order_release);
                    const auto deadline = std::chrono::steady_clock::now() + shutdown_grace_;
                    lock.unlock();
                    stop_accepting();
                    lock.lock();
                    if (!cv_.wait_until(lock, deadline, [this] { return stopping_; })) {
                        // HTTP stop drains workers, including uploads with only an inactivity
                        // timeout. Do not wait for their destructors or buffered I/O to exit.
                        std::_Exit(EXIT_FAILURE);
                    }
                    return;
                }
                cv_.wait_for(lock, std::chrono::milliseconds(100),
                             [this] { return stopping_; });
            }
        });
    }

    void stop() {
        if (!thread_.joinable()) { return; }
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    [[nodiscard]] bool failed() const noexcept {
        return failed_.load(std::memory_order_acquire);
    }

private:
    const std::chrono::milliseconds shutdown_grace_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stopping_ = false;
    std::atomic<bool> failed_{false};
};

} // namespace ninfer::serve
