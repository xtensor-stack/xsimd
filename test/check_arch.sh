#!/bin/sh
set -e
CXX=g++
printf "int main() { return 0;}" > sanity_check.cpp
printf "#include <xsimd/xsimd.hpp>\nint main() { return 0;}" > xsimd_check.cpp

sed -n '/x86[-]64/,$ p' $0 | \
    while read arch; do \
        if echo $arch | grep -q '#' ; then continue; fi ; \
        echo "# $arch" ; \
        $CXX -w -march=$arch sanity_check.cpp -fsyntax-only ;  \
        $CXX -w -I../include -march=$arch xsimd_check.cpp -fsyntax-only ; \
    done

rm sanity_check.cpp xsimd_check.cpp

exit 0
nocona
core2
nehalem
corei7
westmere
sandybridge
corei7-avx
ivybridge
core-avx-i
haswell
core-avx2
broadwell
skylake
skylake-avx512
cannonlake
icelake-client
rocketlake
icelake-server
cascadelake
tigerlake
cooperlake
sapphirerapids
emeraldrapids
alderlake
raptorlake
meteorlake
graniterapids
graniterapids-d
bonnell
atom
silvermont
slm
goldmont
goldmont-plus
tremont
gracemont
sierraforest
grandridge
knl
knm
x86-64
x86-64-v2
x86-64-v3
x86-64-v4
eden-x2
nano
nano-1000
nano-2000
nano-3000
nano-x2
eden-x4
nano-x4
lujiazui
k8
k8-sse3
opteron
opteron-sse3
athlon64
athlon64-sse3
athlon-fx
amdfam10
barcelona
bdver1
bdver2
bdver3
bdver4
znver1
znver2
znver3
znver4
btver1
btver2
