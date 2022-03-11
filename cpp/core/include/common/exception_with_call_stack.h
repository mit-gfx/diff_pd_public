#ifndef COMMON_EXCEPTION_WITH_CALL_STACK_H
#define COMMON_EXCEPTION_WITH_CALL_STACK_H

#include "common/config.h"

class ExceptionWithCallStack : public std::runtime_error {
public:
    explicit ExceptionWithCallStack(const char* message);

    const char* what() const noexcept;
};

#endif