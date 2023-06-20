#!/bin/sh
#
# Usage: $0 top_srcdir
#
# This script walks all headers in $top_srcdir/include and makes sure that all
# functions declared tehre are marked as inline or constexpr (which implies
# inline). This makes sure the xsimd headers does not define symbol with global
# linkage, and somehow convey our itnent to have all functions in xsimd being
# inlined by the compiler.

set -e
set -x

which clang-query || { echo "missing dependency: clang-query" 1>&2 ; exit 1; }

top_srcdir=$1

query_file=`mktemp -t`
sed -r -n '/^####/,$ p' < $0 > $query_file

log_file=`mktemp -t`

clang-query --extra-arg "-std=c++11" --extra-arg="-I$top_srcdir/include" -f $query_file $top_srcdir/include/xsimd/xsimd.hpp -- | tee $log_file

{ grep -E '^0 matches.' $log_file && failed=0 ; } || failed=1

rm -f $query_file
rm -f $log_file

exit $failed

#### clang-query commands ####
set traversal     IgnoreUnlessSpelledInSource
set print-matcher false
set bind-root     false
enable output     diag
match functionDecl(isExpansionInFileMatching(".*/xsimd/.*"), isDefinition(), unless(isInline()), unless(isConstexpr())).bind("inline-function")
