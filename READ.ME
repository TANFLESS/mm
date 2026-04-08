# GitHub 连通性排查备忘（Codex 网页版环境）

你在网页版使用 Codex 时，网络路径通常会经过托管代理层。
如果出现 `CONNECT tunnel failed, response 403`，常见不是仓库权限问题，而是“出站代理策略”拦截。

## 可能原因（按常见度）

1. **平台侧出站代理策略未放通 github.com:443**
   - 现象：`curl` 与 `git ls-remote https://github.com/...` 都报 403。

2. **环境变量里有代理设置，且代理拒绝目标域名**
   - 现象：`env` 中存在 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，请求都走代理。

3. **git 配置层有代理（global/system/local），覆盖了环境设置**
   - 现象：即使清空 env，git 仍通过代理访问。

4. **SSH 通道被封（22 端口不可达）**
   - 现象：`ssh -T git@github.com` 报 `Network is unreachable`。

5. **容器网络策略是“白名单域名”模式**
   - 现象：部分网站可访问，但 GitHub 相关域名被拒。

---

## 命令清单（可直接复制）

### A. 查看当前代理来源

```bash
env | grep -Ei 'http_proxy|https_proxy|all_proxy|no_proxy'
```
用途：查看当前 shell 是否注入代理环境变量。

```bash
git config --show-origin --get-regexp 'http\..*proxy|https\..*proxy'
```
用途：查看 git 在 system/global/local 哪一层配置了代理。

### B. 临时禁用当前 shell 代理并测试连通

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```
用途：仅当前终端会话临时禁用代理变量。

```bash
curl -I https://github.com --max-time 10
```
用途：测试 HTTPS 基础连通性。

```bash
git ls-remote https://github.com/TANFLESS/mm.git HEAD
```
用途：测试 git over HTTPS 到目标仓库是否可达。

### C. 清除 git 代理配置（若有）

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```
用途：删除全局 git 代理配置。

```bash
git config --unset http.proxy
git config --unset https.proxy
```
用途：删除当前仓库级 git 代理配置。

### D. 远端与拉取（目标操作）

```bash
cd /workspace/mm
git remote -v
```
用途：检查 origin 地址是否正确。

```bash
git fetch origin
git branch --set-upstream-to=origin/main work
git pull --rebase
```
用途：拉取远端并设置分支跟踪关系。

### E. 可选：SSH 方案（若 HTTPS 受限）

```bash
ssh -T git@github.com
git remote set-url origin git@github.com:TANFLESS/mm.git
git fetch origin
```
用途：改走 SSH 通道访问仓库。
