/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
#ifndef EMSCRIPTEN
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#else

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#include <emscripten/bind.h>

int run_tests()
{
    doctest::Context context;
    return context.run();
}

EMSCRIPTEN_BINDINGS(my_module)
{
    emscripten::function("run_tests", &run_tests);
}

#endif