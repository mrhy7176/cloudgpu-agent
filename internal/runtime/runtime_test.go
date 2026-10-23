package runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// hub 侧 ApiTunnelInstaller.p2pTunnel 发的就是这些键。两边任何一个改名，隧道就静默起不来。
func TestTunnelFieldsMatchHub(t *testing.T) {
	raw := `{"proxy_name":"0123456789abcdef","custom_domain":"0123456789abcdef-i.cloudgpu.app",
	"server_addr":"43.161.242.247","server_port":7000,"token":"tok","local_port":8888,"bandwidth":"3MB",
	"frpc_urls":["http://a/frpc.gz","http://a/frpc"],"frpc_sha256":"ab"}`
	var tun Tunnel
	if err := json.Unmarshal([]byte(raw), &tun); err != nil {
		t.Fatal(err)
	}
	if tun.ProxyName != "0123456789abcdef" || tun.CustomDomain != "0123456789abcdef-i.cloudgpu.app" ||
		tun.ServerAddr != "43.161.242.247" || tun.ServerPort != 7000 || tun.Token != "tok" ||
		tun.LocalPort != 8888 || tun.Bandwidth != "3MB" || len(tun.FrpcURLs) != 2 || tun.FrpcSHA256 != "ab" {
		t.Fatalf("tunnel fields did not round-trip: %+v", tun)
	}
}

func TestWithoutProxyDropsProxyVars(t *testing.T) {
	out := withoutProxy([]string{"PATH=/bin", "http_proxy=x", "HTTPS_PROXY=y", "ALL_PROXY=z", "HOME=/root"})
	if len(out) != 2 || out[0] != "PATH=/bin" || out[1] != "HOME=/root" {
		t.Fatalf("got %v", out)
	}
}

func TestSHA256Check(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("hello"), 0600); err != nil {
		t.Fatal(err)
	}
	const want = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
	if ok, _ := fileSHA256Matches(p, want); !ok {
		t.Fatal("matching sha rejected")
	}
	if ok, _ := fileSHA256Matches(p, "00"); ok {
		t.Fatal("wrong sha accepted")
	}
	if ok, _ := fileSHA256Matches(filepath.Join(t.TempDir(), "missing"), ""); ok {
		t.Fatal("missing file accepted")
	}
}

func TestRunRefusesWithoutTunnelOrImage(t *testing.T) {
	d := &dockerHandler{claims: map[string]*claim{"c1": {gpuIndex: 0}}}
	if _, err := d.Run(RunSpec{ClaimID: "c1"}); err == nil {
		t.Fatal("run without image should fail")
	}
	if _, err := d.Run(RunSpec{ClaimID: "nope", Image: "x"}); err == nil {
		t.Fatal("run on unknown claim should fail")
	}
}
