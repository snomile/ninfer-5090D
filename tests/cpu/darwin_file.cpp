// Compile the production buffered reader on Darwin for metadata/placement tests.
// Direct I/O is unsupported here and must still be verified on Linux.
#include <cerrno>
#include <fcntl.h>
#define O_DIRECT 0x80000000
#define AT_EMPTY_PATH 0
#define AT_STATX_DONT_SYNC 0
#define STATX_DIOALIGN 1
struct statx {
    unsigned stx_mask = 0, stx_dio_mem_align = 0, stx_dio_offset_align = 0;
};
inline int statx(int, const char*, int, unsigned, struct statx*) {
    errno = EOPNOTSUPP;
    return -1;
}
#include "../../src/core/io/file.cpp"
