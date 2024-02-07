NDK_ROOT ?= $(HOME)/Android/Sdk/ndk/26.1.10909125/
ARCH ?= x86_64
# note: this is 386 for x86 AVD -- retest amd64 on x86_64 AVD + update instructions
GOARCH ?= 386
TARGET_API ?= android28
# onnx version basd on what onnxruntime_go targets
ONNX_VERSION ?= 1.16.1
SO_ARCH ?= x86

droid-mlmodel:
	GOOS=android GOARCH=$(GOARCH) CGO_ENABLED=1 \
		CC=$(shell realpath $(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/$(ARCH)-linux-$(TARGET_API)-clang) \
		go build -v \
		-tags no_cgo \
		-o $@ \
		./cmd/module

onnxruntime-android-$(ONNX_VERSION).aar:
	# see https://onnxruntime.ai/docs/install/#install-on-android
	wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$(ONNX_VERSION)/onnxruntime-android-$(ONNX_VERSION).aar

third_party/onnx-android-$(SO_ARCH).so: onnxruntime-android-$(ONNX_VERSION).aar
	unzip -o $< 'jni/*/*.so'
	cp jni/$(SO_ARCH)/libonnxruntime.so $@

bundle-droid-$(SO_ARCH).tar.gz: droid-mlmodel third_party/onnx-android-$(SO_ARCH).so ir_mobilenet.onnx
	tar -czf $@ $^
