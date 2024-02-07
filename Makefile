NDK_ROOT ?= $(HOME)/Android/Sdk/ndk/26.1.10909125/
ARCH ?= x86_64
# note: this is 386 for x86 AVD -- retest amd64 on x86_64 AVD + update instructions
GOARCH ?= 386
TARGET_API ?= android28

droid-mlmodel:
	GOOS=android GOARCH=$(GOARCH) CGO_ENABLED=1 \
		CC=$(shell realpath $(NDK_ROOT)/toolchains/llvm/prebuilt/linux-x86_64/bin/$(ARCH)-linux-$(TARGET_API)-clang) \
		go build -v \
		-tags no_cgo \
		-o $@ \
		./cmd/module
