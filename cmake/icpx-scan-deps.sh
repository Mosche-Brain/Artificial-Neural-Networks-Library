#!/bin/sh
# Wrapper around clang-scan-deps for icpx (IntelLLVM). The upstream clang-scan-deps does not know some
# Intel specific driver flags, so they are translated/dropped before scanning.
scan_deps="$1"; resource_dir="$2"; shift 3   # drops the "--" separator
compiler="$1"; shift
n=$#
while [ "$n" -gt 0 ]; do
    a="$1"; shift; n=$((n-1))
    case "$a" in
        -fiopenmp) set -- "$@" -fopenmp ;;
        -Rno-*|-fsycl|-fsycl-*|-qopenmp|-qmkl*) ;;
        *) set -- "$@" "$a" ;;
    esac
done
exec "$scan_deps" -format=p1689 -- "$compiler" -resource-dir "$resource_dir" "$@"
