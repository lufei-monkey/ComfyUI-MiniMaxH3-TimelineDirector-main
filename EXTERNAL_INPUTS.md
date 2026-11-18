# 外接素材与分段配置（新版）

新增两个节点：MiniMaxH3ExternalMedia（素材汇总）、MiniMaxH3ExternalSegments（分段汇总）。分别接入 MiniMaxH3TimelinePlanner 的 media_inputs、segment_inputs。导演台仍支持原有直接导入素材；不连接汇总节点时保持原流程。

## 接口
素材汇总：image_0..8 (IMAGE)，ref_video_0..2 (IMAGE 帧批次)，ref_audio_0..2 和 ref_video_audio_0..2 (AUDIO)，global_prompt (STRING)。配对视频音频必须连接相同编号的视频。视频按24 fps解释。

分段汇总：segment_1..64。每段的接口为 segment_N_image_1..9、segment_N_audio_1..3、segment_N_segment_frames、segment_N_segment_prompt、segment_N_overlap_frames，共15个。另保留 segment_N_duration（秒）作为可选时长接口，因此包含时长时每段最多16个口。帧数优先于时长。图片每个接口接一张图。

每组图片、音频接口连接后显示下一个，最多9/3个；任意分段接口连接后显示下一段，最多64段。保存后重载保留已连接接口。新增节点后即可使用，旧版直接连接到导演台的编号接口请迁移到素材/分段汇总节点。

## 分段规则
第一段重叠始终为0。后续段优先使用外接 overlap_frames，否则从导演台原有相邻窗口读取重叠量。帧数必须为5+17*n，最大3592；重叠允许0、1或5+17*n，且必须小于相邻两段帧数，不合法时明确报错。

本段外接图片/音频存在时覆盖该段对应的参考素材分配；未连接则保留导演台分配。全局素材仍追加到已有素材，并受原全局数量限制。外接分段提示词覆盖该段，未连接沿用内部值。全局提示词不是自动拼接到分段提示词：保持原有全局回退规则。Plan Encoder 显式非空 prompt 优先；留空才使用计划提示词。

素材通过内容哈希缓存进入现有文件读取链路。图片PNG、视频FFV1 MKV（8-bit RGB）、音频16-bit PCM WAV；缓存位于 ComfyUI/input/minimax_h3_timeline_director/external，不自动清理。外接配置不改写导演台保存的UI数据，时间轴不会显示新增外接卡片。

## 数据流与验证
汇总节点 → 导演台合并素材与分段窗口 → Material Plan / Omni Media Bundle / Segment Plan → 原 Plan Encoder、有限分段采样或Loop链路。

新增 external_input_nodes.py、js/external_input_nodes.js、tests/test_external_segments_standalone.py；修改主 Python 文件、__init__.py，移除先前直接给导演台动态添加编号接口的JS扩展。

通过：全仓Python语法编译、新JS语法检查、独立分段模块导入和测试、git diff --check。158×3，重叠0/22/39，最终413帧；另测外接优先级、内部回退、首段重叠、非法值、输入不被修改。

限制：本机没有torch、PyAV和ComfyUI，完整插件导入、真实素材编码、浏览器动态接线、保存重载和模型生成尚未实测。需在实际ComfyUI环境验证。安装前备份原目录，替换插件后重启ComfyUI并刷新页面。
