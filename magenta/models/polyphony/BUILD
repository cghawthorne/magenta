# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Model that supports polyphony

py_binary(
    name = "create_dataset",
    srcs = ["create_dataset.py"],
    deps = [
      ":sequence",
      ":one_hot_delta_codec",
      "//magenta/lib:note_sequence_io",
    ]
)

py_binary(
    name = "model",
    srcs = ["model.py"],
    deps = [
      ":sequence",
      ":one_hot_delta_codec",
      "//magenta/lib:note_sequence_io",
      "//magenta/lib:midi_io",
    ]
)

py_test(
    name = "sequence_test",
    srcs = ["sequence_test.py"],
    deps = [
    	":sequence",
	":test_helper",
    ],
)

py_library(
    name = "sequence",
    srcs = ["sequence.py"],
    deps = [
    	"//magenta/protobuf:music_py_pb2"
    ],
)

py_library(
    name = "test_helper",
    srcs = ["test_helper.py"],
    deps = [
    	"//magenta/protobuf:music_py_pb2"
    ],
)

py_library(
    name = "one_hot_delta_codec",
    srcs = ["one_hot_delta_codec.py"],
    deps = [":sequence"],
)

py_test(
    name = "one_hot_delta_codec_test",
    srcs = ["one_hot_delta_codec_test.py"],
    deps = [
    	":sequence",
	":one_hot_delta_codec",
	":test_helper",
    ],
)
