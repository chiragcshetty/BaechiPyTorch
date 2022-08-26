py_library(
    name = "placer_lib",
    srcs = ["placer_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":adjuster",
        ":grouper",
        ":m_etf",
        ":m_etf_sequential",
        ":m_sct",
        ":m_topo",
        ":placer_utils",
        ":virtual_scheduler",
        "//utils:logger",
    ],
)

py_test(
    name = "placer_lib_test",
    srcs = ["placer_lib_test.py"],
    deps = [":placer_lib"],
)

py_library(
    name = "device",
    srcs = ["device.py"],
    deps = [
        "//utils:logger",
    ],
)

py_test(
    name = "device_test",
    srcs = ["device_test.py"],
    deps = [
        ":device",
    ],
)

py_library(
    name = "placer_utils",
    srcs = ["placer_utils.py"],
    deps = [
        "//utils:logger",
    ],
)

py_test(
    name = "placer_utils_test",
    srcs = ["placer_utils_test.py"],
    deps = [":placer_utils"],
)

py_library(
    name = "m_topo",
    srcs = ["m_topo.py"],
    deps = [
        ":device",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "m_etf",
    srcs = ["m_etf.py"],
    deps = [
        ":device",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "m_sct",
    srcs = ["m_sct.py"],
    deps = [
        ":device",
        ":m_etf",
        ":m_sct_v1",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "virtual_scheduler",
    srcs = ["virtual_scheduler.py"],
    deps = ["//utils:logger"],
)

py_library(
    name = "grouper",
    srcs = ["grouper.py"],
    deps = [
        ":placer_utils",
        "//utils:logger",
    ],
)

py_binary(
    name = "grouper_test",
    srcs = ["grouper_test.py"],
    deps = [":grouper"],
)

py_library(
    name = "adjuster",
    srcs = ["adjuster.py"],
    deps = [
        ":placer_utils",
        "//utils:logger",
    ],
)

py_binary(
    name = "adjuster_test",
    srcs = ["adjuster_test.py"],
    deps = [":adjuster"],
)

py_binary(
    name = "run_favorite_child_lp_solver",
    srcs = ["utils/run_favorite_child_lp_solver.py"],
    deps = [
        ":m_sct",
        ":m_sct_v0",
        ":placer_lib",
    ],
)

py_library(
    name = "m_etf_v0",
    srcs = ["deprecated/m_etf_v0.py"],
    deps = [],
)

py_library(
    name = "m_topo_v0",
    srcs = ["deprecated/m_topo_v0.py"],
    deps = ["//utils:logger"],
)

py_library(
    name = "m_sct_v0",
    srcs = ["deprecated/m_sct_v0.py"],
    deps = ["//utils:logger"],
)

py_library(
    name = "m_etf_v1",
    srcs = ["deprecated/m_etf_v1.py"],
    deps = [
        ":device",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_test(
    name = "m_etf_v1_test",
    srcs = ["deprecated/m_etf_v1_test.py"],
    deps = [
        ":m_etf_v1",
        ":placer_lib",
    ],
)

py_library(
    name = "m_sct_v1",
    srcs = ["deprecated/m_sct_v1.py"],
    deps = [
        ":m_etf_v1",
        ":placer_utils",
        "//utils:logger",
    ],
)

py_test(
    name = "m_sct_v1_test",
    srcs = ["deprecated/m_sct_v1_test.py"],
    deps = [
        ":m_sct_v1",
        ":placer_lib",
    ],
)

py_library(
    name = "m_etf_sequential",
    srcs = ["deprecated/m_etf_sequential.py"],
    deps = [
        ":placer_utils",
        "//utils:logger",
    ],
)

py_library(
    name = "cost",
    srcs = ["cost.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:logger",
    ],
)
