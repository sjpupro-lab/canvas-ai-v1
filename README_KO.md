# SPATIAL-PATTERN-AI (CANVAS)

![Main Hero](main_hero.png)

> 텍스트를 256×256 RGBA 격자 위의 밝기 패턴으로 변환하여 학습하는 공간 패턴
> 기반 AI 엔진. 언어를 토큰 벡터가 아닌 영상 프레임처럼 취급합니다 — 매칭,
> 생성, 압축이 모두 이 비트맵에 대한 이미지 연산으로 자연스럽게 떨어집니다.

```
 "귀여운 고양이가 밥을 먹는다."     256×256 격자 (1절 = 1프레임)
         │                        ┌────────────────────────┐
    UTF-8 바이트                    │  ·                     │
         │                        │    ·  ·                 │
   ┌─────▼──────┐                  │  · · ·  ·   ·          │
   │  X = 바이트값│                 │     ·    ·              │
   │  Y = 위치   │   ──────►       │  ·    ·                 │
   │  A = 3레이어 │                 │       ·  ·             │
   │        합계 │                 │  · ·       ·  ·        │
   └─────────────┘                  └────────────────────────┘
                                    A채널: 바이트 빈도 히트맵
```

---

## 목차

- [왜 만들었나](#왜-만들었나)
- [동작 원리](#동작-원리)
  - [1. 3-레이어 비트맵 합산](#1-3-레이어-비트맵-합산)
  - [2. RGBA 채널](#2-rgba-채널)
  - [3. 키프레임 / 델타 저장](#3-키프레임--델타-저장)
  - [4. 매칭 캐스케이드](#4-매칭-캐스케이드)
  - [5. Top-K + 문자 단위 정제](#5-top-k--문자-단위-정제)
  - [6. 캔버스 풀 + 슬롯 재배치](#6-캔버스-풀--슬롯-재배치)
- [빌드 & 실행](#빌드--실행)
- [테스트 & 검증기](#테스트--검증기)
- [저장 / 로드](#저장--로드)
- [프로젝트 구조](#프로젝트-구조)
- [SPEC과의 의도적 차이](#spec과의-의도적-차이)

---

## 왜 만들었나

기존 LLM은 의미를 불투명한 float 벡터로 압축합니다. 다른 모델만이 그 값을
해석할 수 있습니다. 이 엔진은 같은 의미를 **해석 가능한 256×256 이미지**로
압축합니다:

- X축 = 바이트 값 (0–255)
- Y축 = 절 내 바이트 위치
- A채널 = 3-레이어 가중치로 찍힌 "밝기" (빈도)
- R/G/B채널 = 품사(POS) 프라이어가 코퍼스를 걸쳐 EMA로 수렴

비트맵이기 때문에 모든 단계를 **눈으로 볼 수 있습니다**. 그대로 H.264식의
씬체인지 감지와 델타-RLE 압축에 태울 수 있고, 실제로 키프레임/델타와 캔버스
풀 레이어가 그렇게 구성되어 있습니다.

---

## 동작 원리

### 1. 3-레이어 비트맵 합산

![3-Layer Summation](visualization_1.png)

하나의 절을 3장의 독립 비트맵으로 인코딩한 뒤, 투명한 종이를 덧대듯
합산합니다.

| 레이어 | 단위 | 가중치 | 포착하는 것 |
|--------|------|--------|-------------|
| **기본 (Base)** | 모든 바이트 | **+1** | 위치별 바이트 빈도 |
| **단어 (Word)** | 공백 기준 단어 | **+5** | 단어 수준 강조 (치환 가능 단위) |
| **형태소 (Morpheme)** | 사전 형태소 (내용 품사만) | **+3** | 어근/어간 경계 |

```
A_sum(y, x) = base(y, x) + word(y, x) + morpheme(y, x)   (uint16 clamp)
```

형태소 레이어는 **내용 품사(명사/동사/형용사/미분류)만** 찍고, 조사/어미는
기본 가중치에만 기여합니다. 이유는 아래
[SPEC과의 의도적 차이](#spec과의-의도적-차이)에서.

### 2. RGBA 채널

A가 빈도면 R/G/B는 **품사 프라이어**입니다. 3단계 메커니즘이 겹쳐서 동작:

**① POS 시드** (`seed_morpheme_rgb`) — 같은 품사 토큰은 같은 시작색:

| POS | R | G | B |
|-----|---|---|---|
| 명사 NOUN | 40 | 30 | 100 |
| 동사 VERB | 120 | 40 | 140 |
| 형용사 ADJ | 170 | 35 | 180 |
| 조사 PARTICLE | 8 | 85 | 90 |
| 어미 ENDING | 6 | 95 | 110 |
| 구두점 PUNCT | 5 | 120 | 60 |
| 미분류 UNKNOWN | 210 | 20 | 200 |

**② 방향성 확산** (`update_rgb_directional`) — 절 내 활성 셀이 인접 셀 색의
평균쪽으로 당겨짐. 어느 방향으로 당겨지느냐가 어떤 종류의 공출현을 잡는지
결정:

| 채널 | 방향 | 잡는 공출현 |
|------|------|-------------|
| **R** | 대각선 ↗↘↙↖ | 형태소 결합 (어근+조사) |
| **G** | 수직 ↑↓ | 단어 치환 (같은 위치의 대체 가능 단어) |
| **B** | 수평 ←→ | 절 내 바이트 순서 / 슬롯 간 흐름 |

**③ 절 간 EMA** (`ema_update`) — 같은 (y, x) 셀이 여러 절에 걸쳐 활성화되면
그 셀의 R/G/B가 누적 평균으로 수렴. 새 절 인코딩 시 이 평균을 섞어넣음
(`apply_ema_to_grid`). 관측 수 2 미만인 셀은 건드리지 않음.

### 3. 키프레임 / 델타 저장

각 절은 전체 **I-프레임** 또는 희소 **P-프레임(델타)** 중 하나로 저장됩니다.
`ai_store_auto`가 코사인 유사도로 결정:

```
절 → 3레이어 → RGB 확산 → EMA 블렌드
   │
   ├─ topic_bucket_best_match (같은 주제 KF만 선형 스캔, 빠름)
   └─ 유사도 < 0.30 이면 spatial_match(MATCH_SEARCH) 캐스케이드

best_sim ≥ 0.30  →  DeltaFrame(parent_id = best_kf)로 저장
best_sim < 0.30  →  새 Keyframe, bucket_index_add, ema_update
```

임계값은 `ai_set_store_threshold(t)`로 런타임 조정 가능 — 위키처럼 비슷한
문장이 반복되는 코퍼스는 `0.15` 정도가 효과적.

자료구조:

```c
Keyframe   { id, label, SpatialGrid grid, topic_hash, seq_in_topic };
DeltaEntry { index, diff_A (i16), diff_R/G/B (i8) };   // 9바이트, sparse
DeltaFrame { id, parent_id, entries[], count, change_ratio };
```

`apply_delta(base, entries, count, out)`가 셀별로 diff를 더해 타깃 그리드
재구성 (채널별 clamp).

### 4. 매칭 캐스케이드

모든 검색은 `spatial_match` 하나로 통합. 3가지 모드:

| 모드 | 점수 | 용도 |
|------|------|------|
| `MATCH_SEARCH` | 캐스케이드 (겹침 → 코사인) | KF vs 델타 판정 |
| `MATCH_PREDICT` | RGB 가중 코사인 | 다음 프레임 생성 / top-K 검색 |
| `MATCH_GENERATE` | A × B × G 블렌드 | 확장 채널 쌍 검색 |

모든 모드가 같은 `MatchResult { best_id, best_score, topk[8], topk_count }`를
반환. 캐스케이드는 짧은회로 — coarse 16×16 겹침이 이미 임계 미달이면
정밀 RGB 가중 코사인은 생략. `BucketIndex`(활성 셀 앞자리 기반 해시)가
`kf_count`가 커져도 후보 풀을 작게 유지.

### 5. Top-K + 문자 단위 정제

`ai_generate_refine`은 프리픽스로부터 문장을 완성 — top-8 키프레임의
next-in-topic 후속들을 모아 집계한 그리드를 **UTF-8 문자 span 단위로**
순회합니다 (행 단독 argmax 아님).

```
1. 입력 인코딩 → MATCH_PREDICT top-K
2. 각 k: ids_next[k] = ai_next_in_topic(ids[k])
3. focused AggTables 생성 (ids_next의 A/R/G/B 점수 가중 평균)
4. seed = top-1 next-in-topic 그리드 (UTF-8 보장 시작점)
5. refine 루프, 문자 span (1/2/3/4 행) 단위:
     seed span을 agg_score_byte_full로 점수화
     각 top-K 소스의 span 점수화 — UTF-8 폭이 seed와 다르면 기각
     직전 문자와 같은 단어 내부면 같은 소스에 stickiness 보너스
     승리 span을 원자적으로 교체 (source 간 byte 섞임 없음)
6. grid_decode_text_utf8로 디코드
```

이 문자 단위 정제가 한국어 출력을 유효한 한글로 유지하는 핵심. 이전의 행별
argmax는 top-K 간 byte를 섞어 `잌`/`쥸` 같은 Frankenstein 글리프를 만들었음.
`make verify_refine`로 18절 한국어 시드 코퍼스에서 직접 확인 가능.

### 6. 캔버스 풀 + 슬롯 재배치

![Canvas pool](visualization_2.png)

캔버스는 32개 절을 2048×1024 RGBA 버퍼에 **8×4 타일 격자**로 보관
(타일당 256×256):

```
+----+----+----+----+----+----+----+----+
| 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  |
+----+----+----+----+----+----+----+----+
| 8  | 9  | 10 | 11 | 12 | 13 | 14 | 15 |
+----+----+----+----+----+----+----+----+
| 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
+----+----+----+----+----+----+----+----+
| 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 |
+----+----+----+----+----+----+----+----+
```

일반 삽입은 append-only (`canvas_add_clause`는 항상 `slot_count++`).
**캔버스가 가득 찰 때** (slot 32), scene-change 분류 **직전에**
`canvas_reorder_slots`가 greedy 2-opt 패스를 실행:

- 인접 쌍 비용: `(topic_hash 다르면 1 아니면 0) + (1 − A 코사인)`
- 목적함수: 52개 인접 쌍 합 최소화 (28 수평 + 24 수직)
- 물리적 swap은 scratch 버퍼로 cycle-safe하게
- 순열은 callers에 반환 → `SubtitleTrack` 엔트리도
  `subtitle_track_remap_canvas_slots`로 갱신

재배치 후엔 공간적으로 응집된 캔버스가 되어 경계 확산
(`canvas_update_rgb`)이 비슷한 절끼리 서로 강화하고, canvas-level 델타 RLE
압축률도 올라감. 라운드-로빈 최악 배치 테스트에서 측정된 개선:
**인접 쌍 비용 약 32% 감소** (32.24 → 21.79).

재배치 후 캔버스는 씬체인지 감지(16×16 블록합 델타)로 `CANVAS_IFRAME` /
`CANVAS_PFRAME` 분류됨.

---

## 빌드 & 실행

```bash
# 전체 빌드 (라이브러리 + 14개 테스트 + 6개 벤치 + 도구)
make

# 전체 테스트
make test

# 정리
make clean
```

### 한국어 완성 검증기

```bash
make verify_refine     # 18절 한국어 시드, next vs refine 나란히
```

Char-span refine 적용 후 샘플 출력:

```
query   : "고양이가 밥을"
next    : "고양이가 매우 귀엽다."
refine  : "강아지가 밥을 는다다."

query   : "아이가 책을"
refine  : "어양이 책을 읽는다."   ← "읽" 복구
```

### PR 단위 검증기 (base_patterns + wiki5k + 영문)

```bash
make verify_pr2
```

3개 섹션 실행:

- **A**: `data/base_patterns.txt` — top-K 집계가 실제 블렌딩되는지
  (쿼리당 강한 이웃 ≥2) + predicate 복구
- **B**: `data/wiki5k.txt` 회귀 — 1024절 학습, 32개 프리픽스로
  `ai_generate_next` vs `ai_generate_refine` sim + wall time 비교
- **C**: `data/sample_en.txt` — 영문 완성이 ASCII-clean (바이트 ≥0x80 없음)
  → 문자 단위 refine이 script-agnostic임 확인

### 대화형 chat REPL

```bash
./build/chat --train data/wiki5k.txt --max 5000
# 또는 저장된 모델 로드
./build/chat --load build/models/wiki5k.spai
```

### 스트리밍 학습

```bash
./build/stream_train --input data/sample_en.txt --max 50000 \
                     --save build/models/wiki50k.spai --verify
```

중복 많은 코퍼스에선 `--threshold 0.15`로 델타 편향 권장.

### 벤치마크

```bash
make bench
./build/bench_perplexity   data/sample_ko.txt
./build/bench_word_predict data/sample_ko.txt
./build/bench_qa           data/qa.tsv
./build/bench_stsb         data/stsb.tsv
```

---

## 테스트 & 검증기

`make test`로 빌드되는 14개 바이너리:

| 파일 | 검증 범위 |
|------|-----------|
| `test_grid.c` | 256×256 RGBA 기본 연산 |
| `test_layers.c` | 3-레이어 합산, POS 시드 |
| `test_match.c` | 겹침 / 코사인 / 캐스케이드 |
| `test_cascade.c` | `spatial_match` 캐스케이드 배선 |
| `test_keyframe.c` | KF/Delta 저장, 임계값, apply_delta |
| `test_io.c` | `.spai` 파일 I/O 라운드트립 |
| `test_adaptive.c` | 채널 가중치 적응 |
| `test_context.c` | 컨텍스트 프레임 검색 |
| `test_integration.c` | 학습 → 저장 → 매칭 end-to-end |
| `test_canvas.c` | 타일 배치, 델타 RLE, **슬롯 재배치** |
| `test_subtitle.c` | subtitle track + 풀 라우팅 |
| `test_generate_refine.c` | top-K 집계, 패턴 점수, **문자 단위 refine**, stickiness |
| `tools/verify_refine.c` | 한국어 완성 덤프 (18절 × 6주제) |
| `tools/verify_pr2.c` | base_patterns + wiki5k + sample_en 통합 |

---

## 저장 / 로드

```c
int        spatial_ai_save(const SpatialAI* ai, const char* path);
SpatialAI* spatial_ai_load(const char* path);
```

`.spai` 포맷 — tagged trailing-record 레이아웃:

| Tag | 의미 |
|-----|------|
| `0x01` | 키프레임 블록 (id, label, topic_hash, seq_in_topic, 희소 그리드) |
| `0x02` | 델타 블록 (parent_id, entries) |
| `0x03` | 버킷 인덱스 (로드 시 스킵 — 키프레임에서 재구성) |
| `0x04` | 채널 가중치 |
| `0x06` | EMA 테이블 (4 × GRID_TOTAL × float) |

v3 파일을 v4+ 바이너리로 로드 지원 — 누락 레코드는 0으로, 파생 상태는
엔진이 재구성.

---

## 프로젝트 구조

```
include/        공개 헤더
  spatial_grid.h        256×256 RGBA 기본 연산
  spatial_layers.h      3-레이어 인코더 + POS 시드
  spatial_match.h       방향성 확산 + 캐스케이드
  spatial_keyframe.h    SpatialAI, Keyframe, DeltaEntry, EMA
  spatial_generate.h    AggTables, ai_generate_next/refine
  spatial_canvas.h      2048×1024 풀, 32 슬롯, 재배치
  spatial_subtitle.h    SubtitleTrack + SpatialCanvasPool 라우팅
  spatial_morpheme.h    한국어 형태소 분석기
  spatial_io.h          .spai 직렬화
  spatial_context.h     컨텍스트 프레임 검색 (QA)

src/            구현 (헤더당 .c 하나)
tests/          test_* 14개 + bench_* 6개
tools/          chat REPL, stream_train, verify_refine, verify_pr2,
                animate_training.py, visualize_training.py
data/           base_patterns.txt, wiki5k.txt, sample_en.txt
dict/           한국어 형태소 사전 (명사, 동사, 형용사, 조사, 어미)
SPEC.md         전체 아키텍처 명세
SPEC-ENGINE.md  엔진 최적화 명세
```

---

## SPEC과의 의도적 차이

실측 후 의도적으로 `SPEC.md`와 다르게 구현한 부분:

- **레이어 가중치**: SPEC §3.1은 `+1/+2/+1` (기본/단어/형태소). 구현은
  `+1/+5/+3`. 보존 법칙(`A = base + word + morpheme`)은 어느 쪽이든
  성립. 높은 가중치가 단어/형태소 레이어의 코사인 기여를 짧은 코퍼스에서
  키워줌.

- **형태소 레이어 게이트**: SPEC §3.1은 어근/조사/어미를 모두 형태소로
  나열. 구현은 내용 품사(명사/동사/형용사/미분류)만 형태소 레이어에
  찍음 — 조사/어미/구두점은 기본 가중치에만 기여. 기능어를 균일하게
  찍으면 모든 한국어 문장에 공유되는 (가/을/는다/.) 바이트가 증폭되어
  집계에서 내용 신호를 덮어버림. 18절 시드에서 Frankenstein 글리프 복귀로
  실측 검증됨. `pos_is_content`에 주석.

- **Refine 시드**: PR #2 본문은 "row-argmax under RGBA scoring"을 언급
  — 그건 107fd28 이전의 동작이고 byte-mixed 출력을 냈음. 현재 구현은
  top-1 next-in-topic 그리드를 시드로 사용 (항상 UTF-8 유효) + **문자
  span** refine 루프가 top-K 소스를 문자 단위로 집계. `ai_generate_refine`에
  주석.

- **POS_ENDING R 시드** (현재 부합): 이전 R=12 (구체명사 범위
  `[10, 49]` 내). SPEC §5.3의 기능어 범위 `[0, 9]` 준수를 위해 R=6으로
  수정됨.

---

## 라이선스

`LICENSE` 참조.
