# ============================================================
# local_life_agent — Makefile
# ============================================================
# 常用命令:
#   make up       一键启动（构建镜像 + 后台运行）
#   make down     停止并清理
#   make logs     查看实时日志
#   make build    仅构建镜像
#   make clean    停止并清理所有容器资源
#   make shell    进入容器 Shell
# ============================================================

.PHONY: up down logs build clean shell restart

# 一键启动
up:
	docker compose up -d --build

# 停止并移除容器和网络（保留卷数据）
down:
	docker compose down

# 查看实时日志
logs:
	docker compose logs -f

# 仅构建镜像
build:
	docker compose build

# 彻底清理（包括卷数据）
clean:
	docker compose down -v
	-docker rmi local_life_agent:latest

# 进入容器 Shell
shell:
	docker compose exec local-life-agent bash

# 重启服务（不重新构建）
restart:
	docker compose restart
