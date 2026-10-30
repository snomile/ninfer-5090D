#include "core/cpu/system.h"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ninfer {
std::vector<std::thread> cpu_detail::start_workers(std::uint32_t count,
                                                   std::function<void(std::uint32_t)> initialize,
                                                   WorkerBody body, void* argument) {
    if (count == 0 || !initialize || !body)
        throw std::invalid_argument("invalid CPU worker startup");
    struct Startup {
        std::mutex mutex;
        std::condition_variable changed;
        std::uint32_t ready = 0;
        enum class Decision { Pending, Run, Cancel } decision = Decision::Pending;
        std::exception_ptr error;
        std::function<void(std::uint32_t)> initialize;
    };
    auto startup = std::make_shared<Startup>();
    startup->initialize = std::move(initialize);
    std::vector<std::thread> threads;
    threads.reserve(count);
    try {
        for (std::uint32_t rank = 0; rank < count; ++rank) {
            threads.emplace_back([startup, body, argument, rank]() mutable {
                std::exception_ptr error;
                try { startup->initialize(rank); } catch (...) { error = std::current_exception(); }
                bool run;
                {
                    std::unique_lock lock(startup->mutex);
                    if (error && !startup->error) startup->error = error;
                    ++startup->ready;
                    startup->changed.notify_all();
                    startup->changed.wait(
                        lock, [&] { return startup->decision != Startup::Decision::Pending; });
                    run = startup->decision == Startup::Decision::Run;
                }
                startup.reset();
                if (run) body(argument, rank);
            });
        }
        {
            std::unique_lock lock(startup->mutex);
            startup->changed.wait(lock, [&] { return startup->ready == count; });
            if (startup->error) std::rethrow_exception(startup->error);
            startup->decision = Startup::Decision::Run;
        }
        startup->changed.notify_all();
    } catch (...) {
        {
            std::lock_guard lock(startup->mutex);
            startup->decision = Startup::Decision::Cancel;
        }
        startup->changed.notify_all();
        join_workers(threads);
        throw;
    }
    return threads;
}
void cpu_detail::join_workers(std::span<std::thread> workers) noexcept {
    for (auto& worker : workers)
        if (worker.joinable()) worker.join();
}
} // namespace ninfer
