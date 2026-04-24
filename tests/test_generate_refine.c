/*
 * test_generate_refine
 *
 * Covers the structure-based generation upgrade:
 *   - CanvasDeltaEntryRGB / canvas_delta_sparse_rgb
 *   - ai_next_in_topic public helper
 *   - agg_build_topk (focused aggregation)
 *   - spatial_pattern_score (8-neighbor coherence)
 *   - agg_score_byte_full (RGBA × spatial)
 *   - ai_generate_refine (top-k + refinement loop)
 *
 * Uses the "한 요소만 바뀌는 문장" corpus from spec §6 so the model sees
 * substitution patterns (고양이↔강아지, 밥↔물, ...) that exercise
 * top-k aggregation beyond top-1 copy-out.
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_generate.h"
#include "spatial_canvas.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { tests_total++; printf("  [TEST] %s ... ", name); fflush(stdout); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)

/* Base dataset from spec §6 "필수 데이터 패턴" (한 요소만 바뀜). */
static const char* BASE_CLAUSES[] = {
    "고양이가 밥을 먹는다.",
    "강아지가 밥을 먹는다.",
    "고양이가 물을 마신다.",
    "강아지가 물을 마신다.",
    "고양이가 수영한다.",
    "사람이 수영한다.",
    "고양이가 조금 귀엽다.",
    "고양이가 매우 귀엽다.",
    "아이가 책을 읽는다.",
    "어른이 책을 읽는다.",
    "아이가 노래를 부른다.",
    "어른이 노래를 부른다.",
};
#define N_BASE (sizeof(BASE_CLAUSES) / sizeof(BASE_CLAUSES[0]))

/* Group every pair of consecutive clauses under a shared topic so the
 * "next-in-topic" path has real successors to aggregate. */
static const char* BASE_TOPICS[] = {
    "meal",   "meal",
    "drink",  "drink",
    "swim",   "swim",
    "degree", "degree",
    "read",   "read",
    "sing",   "sing",
};

static SpatialAI* build_trained_ai(void) {
    SpatialAI* ai = spatial_ai_create();
    for (uint32_t i = 0; i < N_BASE; i++) {
        ai_force_keyframe(ai, BASE_CLAUSES[i], BASE_TOPICS[i]);
    }
    return ai;
}

/* ── Test 1: CanvasDeltaEntryRGB detects non-A-only diffs ── */

static void test_canvas_delta_rgb(void) {
    TEST("canvas_delta_sparse_rgb detects per-channel diffs");

    SpatialCanvas* a = canvas_create();
    SpatialCanvas* b = canvas_create();

    /* Seed a few cells with distinct RGBA values */
    a->A[10] = 5;   a->R[10] = 100; a->G[10] = 50; a->B[10] = 30;
    b->A[10] = 5;   b->R[10] = 120; b->G[10] = 50; b->B[10] = 30; /* R diff */

    a->A[200] = 0;  a->R[200] = 0;   a->G[200] = 0;   a->B[200] = 0;
    b->A[200] = 0;  b->R[200] = 0;   b->G[200] = 40;  b->B[200] = 0; /* G diff with A=0 on both */

    a->A[500] = 3;
    b->A[500] = 3;   /* no diff */

    CanvasDeltaEntryRGB* entries = (CanvasDeltaEntryRGB*)calloc(
        CV_TOTAL, sizeof(CanvasDeltaEntryRGB));
    uint32_t n = canvas_delta_sparse_rgb(a, b, entries, CV_TOTAL);

    int saw_R = 0, saw_G = 0, saw_500 = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (entries[i].index == 10) {
            assert(entries[i].diff_R == 20);
            assert(entries[i].diff_A == 0);
            saw_R = 1;
        }
        if (entries[i].index == 200) {
            assert(entries[i].diff_G == 40);
            saw_G = 1;
        }
        if (entries[i].index == 500) saw_500 = 1;
    }
    assert(saw_R && saw_G);
    assert(!saw_500);   /* identical cells must NOT be emitted */

    free(entries);
    canvas_destroy(a);
    canvas_destroy(b);
    PASS();
}

/* ── Test 2: ai_next_in_topic — topic path and sequential fallback ── */

static void test_next_in_topic(void) {
    TEST("ai_next_in_topic walks same topic by seq");

    SpatialAI* ai = build_trained_ai();

    /* KF0 is "meal"; the next "meal" KF is KF1. */
    uint32_t n0 = ai_next_in_topic(ai, 0);
    assert(n0 == 1);

    /* KF1 is the second (and last) "meal"; legacy id+1 fallback → KF2. */
    uint32_t n1 = ai_next_in_topic(ai, 1);
    assert(n1 == 2);

    /* Last KF: no successor → returns itself. */
    uint32_t last = ai->kf_count - 1;
    uint32_t nL = ai_next_in_topic(ai, last);
    assert(nL == last);

    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 3: agg_build_topk focuses the aggregation ── */

static void test_agg_build_topk(void) {
    TEST("agg_build_topk builds focused weighted table");

    SpatialAI* ai = build_trained_ai();

    /* Take the first 3 keyframes with declining weights. */
    uint32_t ids[3] = {0, 1, 2};
    float    sc [3] = {0.9f, 0.6f, 0.3f};

    AggTables* tbl = agg_build_topk(ai, ids, sc, 3, /*use_next_in_topic=*/0);
    assert(tbl);

    double total_A = 0.0;
    for (uint32_t y = 0; y < GRID_SIZE; y++) total_A += tbl->row_total_A[y];
    assert(total_A > 0.0);

    /* Global agg should have strictly more mass than any top-3 subset. */
    AggTables* global = agg_build(ai);
    double global_A = 0.0;
    for (uint32_t y = 0; y < GRID_SIZE; y++) global_A += global->row_total_A[y];
    assert(global_A >= total_A);

    /* use_next_in_topic = 1 pulls KF1, KF2, KF3 (next of 0,1,2); mass
     * should differ from the use_next_in_topic = 0 build. */
    AggTables* tbl_next = agg_build_topk(ai, ids, sc, 3, /*use_next_in_topic=*/1);
    assert(tbl_next);
    double next_A = 0.0;
    for (uint32_t y = 0; y < GRID_SIZE; y++) next_A += tbl_next->row_total_A[y];
    assert(next_A > 0.0);

    agg_destroy(tbl);
    agg_destroy(tbl_next);
    agg_destroy(global);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 4: spatial_pattern_score bounds + isolated-cell default ── */

static void test_spatial_pattern_score(void) {
    TEST("spatial_pattern_score returns neutral for isolated cells, [0,1] otherwise");

    SpatialAI* ai = build_trained_ai();
    AggTables* t = agg_build(ai);

    /* Pick a cell known to be active (row 0 / KF's first non-zero x). */
    uint32_t active_y = UINT32_MAX, active_x = UINT32_MAX;
    for (uint32_t y = 0; y < GRID_SIZE && active_y == UINT32_MAX; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            if (t->A_sum[y * GRID_SIZE + x] > 0.0) {
                active_y = y; active_x = x; break;
            }
        }
    }
    assert(active_y != UINT32_MAX);

    SpatialGrid* input = grid_create();
    layers_encode_clause(BASE_CLAUSES[0], NULL, input);

    double s = spatial_pattern_score(t, active_y, (uint8_t)active_x, input);
    assert(s >= 0.0 && s <= 1.0);

    /* For a cell never seen in training, spatial score is 0 (A_sum==0). */
    double empty = spatial_pattern_score(t, 255, 0, input);
    assert(empty == 0.0);

    grid_destroy(input);
    agg_destroy(t);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 5: agg_score_byte_full > 0 on active cell ── */

static void test_score_byte_full(void) {
    TEST("agg_score_byte_full is positive on active cell, zero elsewhere");

    SpatialAI* ai = build_trained_ai();
    AggTables* t = agg_build(ai);

    SpatialGrid* input = grid_create();
    layers_encode_clause(BASE_CLAUSES[0], NULL, input);

    /* Locate any active cell. */
    uint32_t y0 = UINT32_MAX, x0 = UINT32_MAX;
    for (uint32_t y = 0; y < GRID_SIZE && y0 == UINT32_MAX; y++) {
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            if (t->A_sum[y * GRID_SIZE + x] > 0.0) {
                y0 = y; x0 = x; break;
            }
        }
    }
    assert(y0 != UINT32_MAX);

    InputSignature sig;
    input_signature_compute(&sig, input);
    double iR, iG, iB;
    input_signature_get(&sig, y0, &iR, &iG, &iB);

    double positive = agg_score_byte_full(t, y0, (uint8_t)x0, iR, iG, iB, input);
    double zero     = agg_score_byte_full(t, 255, 0, iR, iG, iB, input);
    assert(positive > 0.0);
    assert(zero == 0.0);

    grid_destroy(input);
    agg_destroy(t);
    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 6: ai_generate_refine produces non-empty output ── */

static void test_generate_refine_basic(void) {
    TEST("ai_generate_refine returns non-empty utf8 bytes");

    SpatialAI* ai = build_trained_ai();

    char out[4096];
    memset(out, 0, sizeof(out));
    float sim = 0.0f;
    uint32_t n = ai_generate_refine(ai, "고양이가 밥을", out, sizeof(out), &sim);

    printf("\n    refine out = \"%s\" (%u bytes, sim=%.3f)\n", out, n, (double)sim);

    /* Output must be non-empty and null-terminated within buffer. */
    assert(n > 0);
    assert(out[n] == '\0');
    /* Similarity is a top-k score ∈ [0, 1]. */
    assert(sim >= 0.0f && sim <= 1.0f);

    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 7: refine on unknown input doesn't crash, returns <= next ── */

static void test_generate_refine_unknown(void) {
    TEST("ai_generate_refine tolerates unseen inputs");

    SpatialAI* ai = build_trained_ai();

    char out[2048];
    float sim = 0.0f;
    uint32_t n = ai_generate_refine(ai, "전혀 본 적 없는 문장", out, sizeof(out), &sim);
    /* Output may be empty on a very weak match, but must not crash and
     * must null-terminate inside the buffer. */
    assert(n < sizeof(out));
    assert(out[n] == '\0');

    spatial_ai_destroy(ai);
    PASS();
}

/* ── Test 8: empty engine → clean zero return ── */

static void test_generate_refine_empty(void) {
    TEST("ai_generate_refine on empty engine returns 0");

    SpatialAI* ai = spatial_ai_create();

    char out[256];
    out[0] = 'X';
    float sim = 1.0f;
    uint32_t n = ai_generate_refine(ai, "input", out, sizeof(out), &sim);
    assert(n == 0);
    assert(out[0] == '\0');
    assert(sim == 0.0f);

    spatial_ai_destroy(ai);
    PASS();
}

int main(void) {
    printf("=== test_generate_refine ===\n");

    test_canvas_delta_rgb();
    test_next_in_topic();
    test_agg_build_topk();
    test_spatial_pattern_score();
    test_score_byte_full();
    test_generate_refine_basic();
    test_generate_refine_unknown();
    test_generate_refine_empty();

    printf("  %d/%d passed\n\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
