package runtime

import (
	"reflect"
	"testing"
)

func TestMirrorRefs(t *testing.T) {
	cases := map[string][]string{
		"quay.io/jupyter/pytorch-notebook:cuda12-latest": {"quay.nju.edu.cn/jupyter/pytorch-notebook:cuda12-latest", "quay.m.daocloud.io/jupyter/pytorch-notebook:cuda12-latest"},
		"ollama/ollama:0.5.7":                            {"docker.m.daocloud.io/ollama/ollama:0.5.7"},
		"ubuntu:22.04":                                   {"docker.m.daocloud.io/library/ubuntu:22.04"},
		// 我们自己的仓库没有镜像站
		"cloudgpu.app/public/ollama:2025-12-29-rc1": nil,
		"localhost:5000/x:1":                        nil,
	}
	for img, want := range cases {
		if got := mirrorRefs(img); !reflect.DeepEqual(got, want) {
			t.Errorf("mirrorRefs(%q) = %v, want %v", img, got, want)
		}
	}
}
