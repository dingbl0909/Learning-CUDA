// 修复链接问题的辅助文件
// 提供缺失的 std::__throw_bad_array_new_length() 函数

#include <new>
#include <cstdlib>

namespace std {
    // 提供缺失的符号定义
    // 这个函数在 C++11 中引入，但某些环境下可能缺失
    __attribute__((weak))
    void __throw_bad_array_new_length() {
        // 调用标准的 bad_array_new_length 异常
        throw std::bad_array_new_length();
    }
}
