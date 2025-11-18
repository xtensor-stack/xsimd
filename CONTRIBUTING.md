# Contributing to xsimd

First, thanks for being there! Welcome on board, we will try to make your
contributing journey as good an experience as it can be.

# Submitting patches

Patches should be submitted through Github PR. We did put some effort to setup a
decent Continuous Integration coverage, please try to make it green ;-)

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to keep
the coding style consistent, a ``.clang-format`` file is shipped within the
source, feel free to use it!

# Extending the API

We are open to extending the API, as long as it has been discussed either in an
Issue or a PR. The only constraint is to add testing for new functions, and make
sure they work on all supported architectures, not only your favorite one!

# Licensing

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions. Stated otherwise, there's no copyright
assignment.

# Working locally with Pixi

An optional [Pixi](https://pixi.sh) configuration is available to setup dependencies
and run tasks in Conda environments.
It single executable is easily installable from system package managers or through
direct download.

The most useful task is `test`, which will automatically download packages and re-run
the build steps as needed.
It can run a a mix of configurations with different compilers (configured in different
environments) and target architectures (configured as CMake presets).
For instance, to test the SSE2 environment with the Clang 21 compiler:

```sh
pixi run -e clang-21 test dev-sse2
```

All available environments can be found with ``pixi info`` and are stored in the ``.pixi``
local folder.
All available tasks can be listed with ``pixi task list``.
