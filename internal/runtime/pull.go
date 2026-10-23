package runtime

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// 国内镜像站。2026-10-23 真机实测（江苏的 A100）拉 pytorch-notebook（约 5 GB）：
// quay.io 直连十几分钟一个字节没下来；quay.nju.edu.cn 约 3 MB/s；quay.m.daocloud.io 约 0.1 MB/s（没缓存时回源慢）。
// 所以按快慢顺序先试镜像站，拉完 tag 回原名，再不行才直连原仓库。
// 海外机器上镜像站连不上会很快失败，自动退回原仓库，不受影响。
var registryMirrors = map[string][]string{
	"quay.io":   {"quay.nju.edu.cn", "quay.m.daocloud.io"},
	"docker.io": {"docker.m.daocloud.io"},
}

// 一张大镜像（约 5 GB）在慢线路上也该拉完了；超过这个时间就换下一个源。
const pullTimeout = 60 * time.Minute

// splitRegistry 把镜像名拆成 仓库地址 + 剩下的路径。
// 没写仓库地址的（ubuntu:22.04、ollama/ollama）按 docker 的规则算 docker.io，官方镜像补 library/。
func splitRegistry(img string) (registry, rest string) {
	first, after, found := strings.Cut(img, "/")
	if found && (strings.ContainsAny(first, ".:") || first == "localhost") {
		return first, after
	}
	if !found {
		return "docker.io", "library/" + img
	}
	return "docker.io", img
}

// mirrorRefs 返回这个镜像在各镜像站上的名字；没有镜像站的仓库（比如我们自己的 cloudgpu.app）返回空。
func mirrorRefs(img string) []string {
	reg, rest := splitRegistry(img)
	var out []string
	for _, m := range registryMirrors[reg] {
		out = append(out, m+"/"+rest)
	}
	return out
}

func imagePresent(img string) bool {
	return exec.Command("docker", "image", "inspect", img).Run() == nil
}

func dockerPull(ref string) error {
	ctx, cancel := context.WithTimeout(context.Background(), pullTimeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, "docker", "pull", ref).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, lastLine(string(out)))
	}
	return nil
}

// pullImage 先试国内镜像站，成功就 tag 成原名（之后 docker run 用原名直接命中本地），最后才直连原仓库。
func pullImage(img string) error {
	if imagePresent(img) {
		return nil
	}
	for _, ref := range mirrorRefs(img) {
		start := time.Now()
		if err := dockerPull(ref); err != nil {
			log.Printf("[runtime/docker] pull via mirror %s failed after %s: %v", ref, time.Since(start).Round(time.Second), err)
			continue
		}
		if out, err := exec.Command("docker", "tag", ref, img).CombinedOutput(); err != nil {
			log.Printf("[runtime/docker] tag %s -> %s failed: %v: %s", ref, img, err, lastLine(string(out)))
			continue
		}
		// 只删镜像站那个名字，层还被原名引用着，不会重新下载
		_ = exec.Command("docker", "rmi", ref).Run()
		log.Printf("[runtime/docker] pulled %s via %s in %s", img, ref, time.Since(start).Round(time.Second))
		return nil
	}
	// 直连也重试几次：国内到 cloudgpu.app 的连接会被中途重置（2026-10-23 实测），docker 会续传已下完的层。
	var err error
	for attempt := 1; attempt <= directPullAttempts; attempt++ {
		if err = dockerPull(img); err == nil {
			return nil
		}
		log.Printf("[runtime/docker] pull %s attempt %d/%d failed: %v", img, attempt, directPullAttempts, err)
		time.Sleep(5 * time.Second)
	}
	return err
}

const directPullAttempts = 5

// 正在后台拉的镜像。同一个镜像不并发拉两次（prefetch 和租约同时触发时）。
var (
	pullingMu sync.Mutex
	pulling   = map[string]bool{}
)

// pullTracked 拉镜像并登记「正在拉」；别处已经在拉同一个时返回 false，不重复拉。
func pullTracked(img string) (started bool, err error) {
	pullingMu.Lock()
	if pulling[img] {
		pullingMu.Unlock()
		return false, nil
	}
	pulling[img] = true
	pullingMu.Unlock()
	defer func() {
		pullingMu.Lock()
		delete(pulling, img)
		pullingMu.Unlock()
	}()
	return true, pullImage(img)
}

// pullInBackground 在后台拉镜像；已经在拉的直接返回。
func pullInBackground(img string) {
	go func() {
		if started, err := pullTracked(img); started && err != nil {
			log.Printf("[runtime/docker] background pull %s failed: %v", img, err)
		}
	}()
}
