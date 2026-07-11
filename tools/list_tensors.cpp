#include <cstdio>
#include "gguf.h"
using namespace Laplace;
int main(int argc, char** argv) {
    GGUFContext ctx;
    if (!ctx.open(argv[1])) return 1;
    for (const auto& t : ctx.tensors()) {
        if (t.name.find("blk.0.") == 0 || t.name.find("blk.1.") == 0)
            printf("%-55s [%llu,%llu] %s\n", t.name.c_str(), (unsigned long long)t.dims[1], (unsigned long long)t.dims[0], type_name(t.type));
    }
    return 0;
}
