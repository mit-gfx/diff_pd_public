#include "common/exception_with_call_stack.h"
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
#include "common/common.h"

/** Print a demangled stack backtrace of the caller function to FILE* out. */
static inline void PrintStacktrace() {
    // https://panthema.net/2008/0901-stacktrace-demangled/.
    std::cout << RedHead();
    std::cout << "Stack trace:" << std::endl;

    const int max_frames = 16;
    void* addr_list[max_frames];
    const int addr_len = backtrace(addr_list, max_frames);
    if (addr_len == 0) {
        std::cout << "  <empty, possibly corrupt>" << RedTail() << std::endl;
        return;
    }

    // Resolve addresses into strings containing "filename(function+address)", this array must be free()-ed.
    char** symbol_list = backtrace_symbols(addr_list, addr_len);

    // allocate string which will be filled with the demangled function name
    size_t func_name_size = 256;
    char* func_name = static_cast<char*>(malloc(func_name_size));

    // Iterate over the returned symbol lines. Skip the first, it is the
    // address of this function.
    for (int i = 1; i < addr_len; i++) {
        char* begin_name = nullptr;
        char* begin_offset = nullptr;
        char* end_offset = nullptr;
        // Find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = symbol_list[i]; *p; ++p) {
            if (*p == '(') begin_name = p;
            else if (*p == '+') begin_offset = p;
            else if (*p == ')' && begin_offset) {
                end_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // Mangled name is now in [begin_name, begin_offset) and caller offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():
            int status;
            char* ret = abi::__cxa_demangle(begin_name, func_name, &func_name_size, &status);
            if (status == 0) {
                func_name = ret; // Use possibly realloc()-ed string.
                std::cout << "  " << std::string(symbol_list[i]) << " : " << std::string(func_name)
                    + "+" + std::string(begin_offset) << std::endl;
            } else {
                // Demangling failed. Output function name as a C function with no arguments.
                std::cout << "  " << std::string(symbol_list[i]) << " : " << std::string(begin_name)
                    + "()+" + std::string(begin_offset) << std::endl;
            }
        } else {
            // Couldn't parse the line? Print the whole line.
            std::cout << "  " << std::string(symbol_list[i]) << std::endl;
        }
    }

    free(func_name);
    free(symbol_list);

    std::cout << RedTail() << std::endl;
    return;
}

ExceptionWithCallStack::ExceptionWithCallStack(const char* message)
    : std::runtime_error(message) {}

const char* ExceptionWithCallStack::what() const noexcept {
    PrintStacktrace();
    return std::runtime_error::what();
}
