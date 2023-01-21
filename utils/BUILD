py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "session_hooks",
    srcs = ["session_hooks.py"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_binary(
    name = "parse_step_stats",
    srcs = ["parse_step_stats.py"],
)

py_binary(
    name = "parse_timeline",
    srcs = ["parse_timeline.py"],
)

py_library(
    name = "parse_timeline_lib",
    srcs = ["parse_timeline.py"],
)

py_binary(
    name = "device_proto_generator",
    srcs = ["device_proto_generator.py"]
)

py_binary(
    name = "measure_wait_times",
    srcs = ["measure_wait_times.py"],
    deps = [
        ":logger",
        ":parse_timeline_lib",
    ],
)

py_binary(
    name = "parse_op_events_in_timeline",
    srcs = ["parse_op_events_in_timeline.py"],
    deps = [
        ":parse_timeline_lib",
    ],
)
