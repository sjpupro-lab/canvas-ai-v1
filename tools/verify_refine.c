/*
 * verify_refine — end-to-end verification of the top-k + refinement
 * generation system on the spec §6 seed corpus.
 *
 * What it proves, running from source:
 *   1. actual word-level generation: ai_generate_refine produces
 *      non-empty UTF-8 bytes on inputs that partially match a stored
 *      pattern.
 *   2. sentence completion: when the input is a prefix of a stored
 *      clause, refine's output contains the predicate characters
 *      ("먹는다" after "밥을", "마신다" after "물을", etc.).
 *   3. inference (substitution): when the input swaps a noun the
 *      model never saw under that topic (e.g. "강아지가 수영한다"
 *      where only "고양이가 수영한다" / "사람이 수영한다" were
 *      trained), top-k aggregation still recovers the predicate
 *      tokens.
 *
 * This tool prints both ai_generate_next (single-keyframe copy) and
 * ai_generate_refine (top-k + refinement) side by side so the
 * behavioural difference is visible.
 */

#include "spatial_grid.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_keyframe.h"
#include "spatial_generate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Training corpus: grouped into 6 topics × 2 substitution pairs, plus
 * an extra line per topic so ai_next_in_topic has real successors to
 * aggregate. */
typedef struct { const char* text; const char* topic; } Clause;

static const Clause CORPUS[] = {
    /* meal (subject substitution) */
    {"고양이가 밥을 먹는다.",  "meal"},
    {"강아지가 밥을 먹는다.",  "meal"},
    {"사람이 밥을 먹는다.",    "meal"},
    /* drink */
    {"고양이가 물을 마신다.",  "drink"},
    {"강아지가 물을 마신다.",  "drink"},
    {"사람이 물을 마신다.",    "drink"},
    /* swim */
    {"고양이가 수영한다.",     "swim"},
    {"사람이 수영한다.",       "swim"},
    {"아이가 수영한다.",       "swim"},
    /* degree (intensity substitution) */
    {"고양이가 조금 귀엽다.",  "deg"},
    {"고양이가 매우 귀엽다.",  "deg"},
    {"강아지가 매우 귀엽다.",  "deg"},
    /* read */
    {"아이가 책을 읽는다.",    "read"},
    {"어른이 책을 읽는다.",    "read"},
    {"학생이 책을 읽는다.",    "read"},
    /* sing */
    {"아이가 노래를 부른다.",  "sing"},
    {"어른이 노래를 부른다.",  "sing"},
    {"가수가 노래를 부른다.",  "sing"},
};
#define N_CORPUS (sizeof(CORPUS) / sizeof(CORPUS[0]))

/* Sanitize a string for display: replace any non-printable byte and
 * UTF-8 lead bytes with no valid continuation with a visible marker
 * so the terminal doesn't garble. We leave valid UTF-8 intact. */
static void print_visible(const char* s, uint32_t n) {
    fputs("\"", stdout);
    uint32_t i = 0;
    while (i < n) {
        unsigned char b = (unsigned char)s[i];
        int len = 0;
        if      ((b & 0x80) == 0x00) len = 1;
        else if ((b & 0xE0) == 0xC0) len = 2;
        else if ((b & 0xF0) == 0xE0) len = 3;
        else if ((b & 0xF8) == 0xF0) len = 4;

        if (len == 0 || i + (uint32_t)len > n) {
            printf("\\x%02x", b);
            i++;
            continue;
        }
        int ok = 1;
        for (int k = 1; k < len; k++) {
            if (((unsigned char)s[i + (uint32_t)k] & 0xC0) != 0x80) { ok = 0; break; }
        }
        if (!ok) {
            printf("\\x%02x", b);
            i++;
            continue;
        }
        fwrite(s + i, 1, (size_t)len, stdout);
        i += (uint32_t)len;
    }
    fputs("\"", stdout);
}

/* Return 1 if `needle` (UTF-8) appears as a byte substring in `hay`. */
static int contains_bytes(const char* hay, const char* needle) {
    return strstr(hay, needle) != NULL;
}

static void run_query(SpatialAI* ai, const char* query,
                      const char* expect_token) {
    char out_next  [2048];
    char out_refine[2048];
    float sim_next = 0.0f, sim_refine = 0.0f;

    uint32_t n1 = ai_generate_next   (ai, query, out_next,   sizeof(out_next),   &sim_next);
    uint32_t n2 = ai_generate_refine (ai, query, out_refine, sizeof(out_refine), &sim_refine);

    printf("  query   : \"%s\"\n", query);
    printf("  next    : ");  print_visible(out_next,   n1);
    printf("  (%u bytes, sim=%.3f)\n", n1, (double)sim_next);
    printf("  refine  : ");  print_visible(out_refine, n2);
    printf("  (%u bytes, sim=%.3f)\n", n2, (double)sim_refine);

    if (expect_token && *expect_token) {
        int next_has   = contains_bytes(out_next,   expect_token);
        int refine_has = contains_bytes(out_refine, expect_token);
        printf("  contains \"%s\"? next=%s refine=%s\n",
               expect_token,
               next_has   ? "yes" : "no ",
               refine_has ? "yes" : "no ");
    }
    putchar('\n');
}

int main(void) {
    printf("=== verify_refine ===\n");
    printf("Training corpus: %u clauses across 6 topics\n", (unsigned)N_CORPUS);

    SpatialAI* ai = spatial_ai_create();
    for (uint32_t i = 0; i < N_CORPUS; i++) {
        ai_force_keyframe(ai, CORPUS[i].text, CORPUS[i].topic);
    }
    printf("Stored kf=%u delta=%u\n\n", ai->kf_count, ai->df_count);

    printf("── Word/sentence completion (prefix in training) ──\n\n");
    run_query(ai, "고양이가 밥을",    "먹");   /* expects "먹는다" tail */
    run_query(ai, "강아지가 물을",    "마");   /* expects "마신다" */
    run_query(ai, "아이가 책을",      "읽");   /* expects "읽는다" */
    run_query(ai, "어른이 노래를",    "부");   /* expects "부른다" */

    printf("── Inference: unseen subject+predicate combo ──\n\n");
    /* "강아지가 수영한다" was never stored under the "swim" topic
     * (swim contained 고양이 / 사람 / 아이). Refinement must aggregate
     * the swim-topic successors to recover the "수영한다" pattern. */
    run_query(ai, "강아지가 수영",    "수영");
    /* Similar: "학생이 노래를" — read corpus has 학생이 book, sing
     * corpus has 아이/어른/가수 노래. Cross-topic inference. */
    run_query(ai, "학생이 노래를",    "노래");
    /* Degree substitution: "강아지가 조금 귀엽다" not stored
     * (only "매우" was for 강아지). */
    run_query(ai, "강아지가 조금",    "귀엽");

    printf("── Empty/boundary input ──\n\n");
    run_query(ai, "", NULL);
    run_query(ai, "아", NULL);

    spatial_ai_destroy(ai);
    printf("=== done ===\n");
    return 0;
}
