package gpu

import "testing"

func TestMemoryGB(t *testing.T) {
	if MemoryGB() <= 0 {
		t.Fatal("MemTotal not read from /proc/meminfo")
	}
}

func TestScanWithoutNvidiaSmiIsEmptyNotNil(t *testing.T) {
	d, err := Scan()
	if err == nil && d == nil {
		t.Fatal("Scan returned nil slice; hub would record null instead of 0 GPUs")
	}
}
