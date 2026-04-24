#include "spatial_generate.h"
#include "spatial_layers.h"
#include "spatial_morpheme.h"
#include "spatial_match.h"
#include "spatial_context.h"
#include "spatial_canvas.h"
#include "spatial_subtitle.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Aggregated tables ─────────────────────────────────── */

AggTables* agg_build(const SpatialAI* ai) {
    if (!ai) return NULL;
    AggTables* t = (AggTables*)calloc(1, sizeof(AggTables));
    if (!t) return NULL;

    /* Sum A; accumulate A-weighted sums of R, G, B per (y, x) */
    for (uint32_t k = 0; k < ai->kf_count; k++) {
        const SpatialGrid* g = &ai->keyframes[k].grid;
        for (uint32_t i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
            uint16_t a = g->A[i];
            if (a == 0) continue;
            double da = (double)a;
            t->A_sum [i] += da;
            t->R_mean[i] += da * (double)g->R[i];
            t->G_mean[i] += da * (double)g->G[i];
            t->B_mean[i] += da * (double)g->B[i];
        }
    }

    /* Finalize: divide weighted sums by A_sum to get means;
       compute per-row activation totals. */
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (t->A_sum[i] > 0.0) {
                t->R_mean[i] /= t->A_sum[i];
                t->G_mean[i] /= t->A_sum[i];
                t->B_mean[i] /= t->A_sum[i];
            }
            row += t->A_sum[i];
        }
        t->row_total_A[y] = row;
    }
    return t;
}

AggTables* agg_build_from_pool(const struct SpatialCanvasPool_* pool) {
    if (!pool) return NULL;
    AggTables* t = (AggTables*)calloc(1, sizeof(AggTables));
    if (!t) return NULL;

    /* Iterate every populated slot in every canvas, aggregating into
     * tile-local (y, x) coordinates. This mirrors agg_build but with
     * pool as the source of training patterns. */
    for (uint32_t ei = 0; ei < pool->track.count; ei++) {
        const SubtitleEntry* e = &pool->track.entries[ei];
        const SpatialCanvas* c = pool->canvases[e->canvas_id];
        uint32_t x0, y0;
        canvas_slot_byte_offset(e->slot_id, &x0, &y0);

        for (uint32_t dy = 0; dy < GRID_SIZE; dy++) {
            for (uint32_t dx = 0; dx < GRID_SIZE; dx++) {
                uint32_t ti = dy * GRID_SIZE + dx;
                uint32_t ci = (y0 + dy) * CV_WIDTH + (x0 + dx);
                uint16_t a = c->A[ci];
                if (a == 0) continue;
                double da = (double)a;
                t->A_sum [ti] += da;
                t->R_mean[ti] += da * (double)c->R[ci];
                t->G_mean[ti] += da * (double)c->G[ci];
                t->B_mean[ti] += da * (double)c->B[ci];
            }
        }
    }

    /* Finalise means */
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (t->A_sum[i] > 0.0) {
                t->R_mean[i] /= t->A_sum[i];
                t->G_mean[i] /= t->A_sum[i];
                t->B_mean[i] /= t->A_sum[i];
            }
            row += t->A_sum[i];
        }
        t->row_total_A[y] = row;
    }
    return t;
}

void agg_destroy(AggTables* t) { free(t); }

/* ── Input signature ────────────────────────────────────── */

void input_signature_compute(InputSignature* sig, const SpatialGrid* input) {
    if (!sig || !input) return;
    memset(sig, 0, sizeof(*sig));

    double global_aw = 0.0, global_rw = 0.0, global_gw = 0.0, global_bw = 0.0;

    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double aw = 0.0, rw = 0.0, gw = 0.0, bw = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (input->A[i] == 0) continue;
            double da = (double)input->A[i];
            aw += da;
            rw += da * (double)input->R[i];
            gw += da * (double)input->G[i];
            bw += da * (double)input->B[i];
        }
        if (aw > 0.0) {
            sig->R_row[y] = rw / aw;
            sig->G_row[y] = gw / aw;
            sig->B_row[y] = bw / aw;
            sig->has_activity[y] = 1;
        }
        global_aw += aw;
        global_rw += rw;
        global_gw += gw;
        global_bw += bw;
    }

    if (global_aw > 0.0) {
        sig->R_global = global_rw / global_aw;
        sig->G_global = global_gw / global_aw;
        sig->B_global = global_bw / global_aw;
    }
}

void input_signature_get(const InputSignature* sig, uint32_t y,
                         double* out_R, double* out_G, double* out_B) {
    if (!sig || !out_R || !out_G || !out_B) return;

    /* Fast path: this row has activity */
    if (sig->has_activity[y]) {
        *out_R = sig->R_row[y];
        *out_G = sig->G_row[y];
        *out_B = sig->B_row[y];
        return;
    }

    /* Fallback: nearest active neighbor row within a window */
    for (int d = 1; d < 32; d++) {
        int yu = (int)y - d;
        int yd = (int)y + d;
        if (yu >= 0 && sig->has_activity[yu]) {
            *out_R = sig->R_row[yu];
            *out_G = sig->G_row[yu];
            *out_B = sig->B_row[yu];
            return;
        }
        if (yd < (int)GRID_SIZE && sig->has_activity[yd]) {
            *out_R = sig->R_row[yd];
            *out_G = sig->G_row[yd];
            *out_B = sig->B_row[yd];
            return;
        }
    }

    /* Last resort: global clause signature */
    *out_R = sig->R_global;
    *out_G = sig->G_global;
    *out_B = sig->B_global;
}

/* ── Byte scoring: A × G_sim × R_sim ──────────────────── */

double agg_score_byte(const AggTables* t, uint32_t y, uint8_t v,
                      double in_R, double in_G, double in_B) {
    if (!t) return 0.0;
    uint32_t i = y * GRID_SIZE + (uint32_t)v;
    double A = t->A_sum[i];
    if (A <= 0.0) return 0.0;

    double R = t->R_mean[i];
    double G = t->G_mean[i];
    double B = t->B_mean[i];

    double R_sim = 1.0 - fabs(R - in_R) / 255.0;
    double G_sim = 1.0 - fabs(G - in_G) / 255.0;
    double B_sim = 1.0 - fabs(B - in_B) / 255.0;
    if (R_sim < 0.0) R_sim = 0.0;
    if (G_sim < 0.0) G_sim = 0.0;
    if (B_sim < 0.0) B_sim = 0.0;

    /* Full A × R × G × B product — SPEC §5.1 §9.4 */
    return A * R_sim * G_sim * B_sim;
}

/* ── Grid → text decoding ───────────────────────────────
 *
 * Two variants live side by side:
 *
 *   grid_decode_text       pure row-argmax (legacy). Fast, byte-level,
 *                          no UTF-8 awareness. Kept for callers that
 *                          feed ASCII or don't care about multi-byte
 *                          integrity (e.g. bench_qa byte snapshots).
 *
 *   grid_decode_text_utf8  UTF-8 aware. Validates lead + continuation
 *                          bytes across consecutive rows so Korean and
 *                          other multi-byte output doesn't clip. Used
 *                          by ai_generate_next.
 *
 * Both read row-by-row and stop at the first empty row.
 */

static int utf8_lead_len(uint8_t b) {
    if ((b & 0x80) == 0x00) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

static int utf8_is_cont(uint8_t b) { return (b & 0xC0) == 0x80; }

/* Fill out_bytes with up to n x-candidates sorted by A descending.
 * Missing slots get A=0 sentinels. */
static void row_top_n(const SpatialGrid* g, uint32_t y,
                      uint8_t* out_bytes, uint16_t* out_scores, int n) {
    for (int i = 0; i < n; i++) { out_bytes[i] = 0; out_scores[i] = 0; }
    for (uint32_t x = 0; x < GRID_SIZE; x++) {
        uint16_t a = g->A[y * GRID_SIZE + x];
        if (a == 0) continue;
        for (int k = 0; k < n; k++) {
            if (a > out_scores[k]) {
                for (int j = n - 1; j > k; j--) {
                    out_bytes[j]  = out_bytes[j - 1];
                    out_scores[j] = out_scores[j - 1];
                }
                out_bytes[k]  = (uint8_t)x;
                out_scores[k] = a;
                break;
            }
        }
    }
}

/* Legacy: row-argmax, one byte per row, no UTF-8 validation. */
uint32_t grid_decode_text(const SpatialGrid* g, char* out, uint32_t max_out) {
    if (!g || !out || max_out == 0) return 0;

    uint32_t written = 0;
    for (uint32_t y = 0; y < GRID_SIZE && written + 1 < max_out; y++) {
        uint32_t best_x = 0;
        uint16_t best_a = 0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (g->A[i] > best_a) {
                best_a = g->A[i];
                best_x = x;
            }
        }
        if (best_a == 0) break;
        out[written++] = (char)(uint8_t)best_x;
    }
    out[written] = '\0';
    return written;
}

/* UTF-8 aware: row-argmax for the lead byte, then consume
 * `utf8_lead_len(lead) - 1` continuation bytes from the following
 * rows. If the required continuations aren't present in the top
 * candidates, fall back to a single-byte emit so ASCII still round-
 * trips and garbled cells don't stall the decoder. */
uint32_t grid_decode_text_utf8(const SpatialGrid* g, char* out, uint32_t max_out) {
    if (!g || !out || max_out == 0) return 0;

    uint32_t written = 0;
    uint32_t y = 0;

    while (y < GRID_SIZE && written + 4 < max_out) {
        uint8_t  cands[4];
        uint16_t scores[4];
        row_top_n(g, y, cands, scores, 4);

        if (scores[0] == 0) break;  /* empty row = clause end */

        uint8_t lead = cands[0];
        int len = utf8_lead_len(lead);

        if (len == 1) {
            out[written++] = (char)lead;
            y++;
            continue;
        }
        if (len == 0) {
            /* stray continuation or invalid lead — keep legacy behavior:
             * emit the raw byte and advance. */
            out[written++] = (char)lead;
            y++;
            continue;
        }

        /* multi-byte: look for continuation bytes on the next rows */
        uint8_t seq[4] = { lead, 0, 0, 0 };
        int ok = 1;
        for (int k = 1; k < len; k++) {
            if (y + (uint32_t)k >= GRID_SIZE) { ok = 0; break; }
            uint8_t  nb[4];
            uint16_t ns[4];
            row_top_n(g, y + (uint32_t)k, nb, ns, 4);
            int found = 0;
            for (int c = 0; c < 4; c++) {
                if (ns[c] == 0) break;
                if (utf8_is_cont(nb[c])) { seq[k] = nb[c]; found = 1; break; }
            }
            if (!found) { ok = 0; break; }
        }

        if (ok && written + (uint32_t)len < max_out) {
            for (int k = 0; k < len; k++) out[written++] = (char)seq[k];
            y += (uint32_t)len;
        } else {
            /* Validation failed: fall back to single-byte emit so ASCII
             * still round-trips. */
            out[written++] = (char)lead;
            y++;
        }
    }

    out[written] = '\0';
    return written;
}

/* ── Top-K weighted AggTables ─────────────────────────────
 *
 * agg_build / agg_build_from_pool aggregate over *every* keyframe or
 * canvas slot — a global prior. For refinement generation we want a
 * focused prior: only the top-k most similar frames, optionally their
 * next-in-topic successors, each weighted by its match score so the
 * strongest match dominates the aggregated R/G/B.
 */
AggTables* agg_build_topk(const SpatialAI* ai,
                          const uint32_t* ids, const float* scores,
                          uint32_t count, int use_next_in_topic) {
    if (!ai || !ids || count == 0) return NULL;
    AggTables* t = (AggTables*)calloc(1, sizeof(AggTables));
    if (!t) return NULL;

    for (uint32_t c = 0; c < count; c++) {
        uint32_t src = ids[c];
        if (use_next_in_topic) src = ai_next_in_topic(ai, src);
        if (src >= ai->kf_count) continue;

        /* Floor keeps every top-k contributor alive even when the raw
         * match score came back at 0 (e.g. no B-channel activity on a
         * new corpus). Without the floor the strongest match would
         * monopolise the aggregation and we'd collapse back to top-1. */
        double w = scores ? (double)scores[c] : 1.0;
        if (!(w > 0.05)) w = 0.05;

        const SpatialGrid* g = &ai->keyframes[src].grid;
        for (uint32_t i = 0; i < GRID_TOTAL; i++) {
            uint16_t a = g->A[i];
            if (a == 0) continue;
            double da = (double)a * w;
            t->A_sum [i] += da;
            t->R_mean[i] += da * (double)g->R[i];
            t->G_mean[i] += da * (double)g->G[i];
            t->B_mean[i] += da * (double)g->B[i];
        }
    }

    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        double row = 0.0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (t->A_sum[i] > 0.0) {
                t->R_mean[i] /= t->A_sum[i];
                t->G_mean[i] /= t->A_sum[i];
                t->B_mean[i] /= t->A_sum[i];
            }
            row += t->A_sum[i];
        }
        t->row_total_A[y] = row;
    }
    return t;
}

/* ── Spatial pattern similarity ────────────────────────── */

double spatial_pattern_score(const AggTables* t, uint32_t y, uint8_t v,
                             const SpatialGrid* input) {
    if (!t) return 0.5;
    uint32_t ci = y * GRID_SIZE + (uint32_t)v;
    if (t->A_sum[ci] <= 0.0) return 0.0;

    double cR = t->R_mean[ci];
    double cG = t->G_mean[ci];
    double cB = t->B_mean[ci];

    /* 8-neighbor offsets: up/down/left/right + 4 diagonals */
    static const int offs[8][2] = {
        {-1,-1},{-1, 0},{-1, 1},
        { 0,-1},        { 0, 1},
        { 1,-1},{ 1, 0},{ 1, 1}
    };

    double coh_sum = 0.0;  int coh_n = 0;
    double inp_sum = 0.0;  int inp_n = 0;

    for (int k = 0; k < 8; k++) {
        int ny = (int)y + offs[k][0];
        int nv = (int)v + offs[k][1];
        if (ny < 0 || ny >= (int)GRID_SIZE ||
            nv < 0 || nv >= (int)GRID_SIZE) continue;
        uint32_t ni = (uint32_t)(ny * (int)GRID_SIZE + nv);

        /* Cluster coherence: AggTable neighbor's color vs center color */
        if (t->A_sum[ni] > 0.0) {
            double nR = t->R_mean[ni];
            double nG = t->G_mean[ni];
            double nB = t->B_mean[ni];
            double dR = fabs(nR - cR) / 255.0;
            double dG = fabs(nG - cG) / 255.0;
            double dB = fabs(nB - cB) / 255.0;
            double sim = 1.0 - (dR + dG + dB) / 3.0;
            if (sim < 0.0) sim = 0.0;
            coh_sum += sim;
            coh_n++;
        }

        /* Input agreement: input's neighbor color vs center color */
        if (input && input->A[ni] > 0) {
            double dR = fabs((double)input->R[ni] - cR) / 255.0;
            double dG = fabs((double)input->G[ni] - cG) / 255.0;
            double dB = fabs((double)input->B[ni] - cB) / 255.0;
            double sim = 1.0 - (dR + dG + dB) / 3.0;
            if (sim < 0.0) sim = 0.0;
            inp_sum += sim;
            inp_n++;
        }
    }

    double coh = (coh_n > 0) ? (coh_sum / (double)coh_n) : 0.5;
    double inp = (inp_n > 0) ? (inp_sum / (double)inp_n) : 0.5;
    /* Equal weight: cluster prior + input grounding. */
    return 0.5 * coh + 0.5 * inp;
}

double agg_score_byte_full(const AggTables* t, uint32_t y, uint8_t v,
                           double in_R, double in_G, double in_B,
                           const SpatialGrid* input) {
    double base = agg_score_byte(t, y, v, in_R, in_G, in_B);
    if (base <= 0.0) return 0.0;
    double sp = spatial_pattern_score(t, y, v, input);
    /* Smooth to [0.5, 1.0]: keeps the RGBA product as the dominant
     * signal and treats spatial_pattern as a modulator rather than a
     * veto. Matches spec §4.4 "spatial_weight × delta_weight". */
    double modulator = 0.5 + 0.5 * sp;
    return base * modulator;
}

/* ── Full-clause generation ────────────────────────────── */

/* Topic-aware next-frame lookup lives in spatial_keyframe.c so it can
 * be shared with refinement generation; we just use ai_next_in_topic. */

uint32_t ai_generate_next(SpatialAI* ai, const char* input_text,
                          char* out, uint32_t max_out,
                          float* out_match_similarity) {
    if (!ai || !input_text || !out || max_out == 0 || ai->kf_count == 0) {
        if (out && max_out > 0) out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 1. Encode input through full pipeline */
    SpatialGrid* in_grid = grid_create();
    layers_encode_clause(input_text, NULL, in_grid);
    update_rgb_directional(in_grid);
    apply_ema_to_grid(ai, in_grid);

    /* 2. Unified match in GENERATE mode (bg_score precision stage).
     *    With the B channel now POS-seeded (spec v2 Mod D), MATCH_GENERATE
     *    favors candidates whose B × G pattern matches the query, which
     *    is a better proxy for "what should come next" than a pure
     *    cosine. The engine's bucket index is passed through so
     *    large-corpus retrieval stays fast. */
    MatchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.bucket_idx = &ai->bucket_idx;
    MatchResult r = spatial_match(ai, in_grid, MATCH_GENERATE, &ctx);
    grid_destroy(in_grid);

    if (out_match_similarity) *out_match_similarity = r.best_score;
    if (r.best_id >= ai->kf_count) {
        out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 3. Next frame: topic-aware if the matched KF has a topic tag,
     *    otherwise sequential (legacy). */
    uint32_t target_id = ai_next_in_topic(ai, r.best_id);

    /* 4. Decode target frame's grid → text (UTF-8 aware). */
    return grid_decode_text_utf8(&ai->keyframes[target_id].grid, out, max_out);
}

/* ── Character-level refinement helpers ──────────────────
 *
 * These three helpers exist so ai_generate_refine can swap entire
 * UTF-8 character spans atomically instead of mixing bytes across
 * sources on a row-by-row basis (which produced "사잌이"-style
 * Frankenstein glyphs on the 18-clause seed corpus).
 */

/* Fill out_bytes[k][y] = argmax_x A[y,x] for each source k's grid.
 * These bytes form each source's "canonical" output byte stream; the
 * refine loop walks them in UTF-8 character spans. Sources whose grid
 * has no activity at row y leave out_bytes[k][y] = 0 (which
 * utf8_lead_len maps to role 0 / ASCII-NUL and is rejected by the
 * span validator). */
static void refine_collect_source_bytes(
        const SpatialAI* ai,
        const uint32_t* ids_next, uint32_t kcount,
        uint8_t out_bytes[TOP_K][GRID_SIZE]) {
    for (uint32_t k = 0; k < kcount && k < TOP_K; k++) {
        uint32_t sid = ids_next[k];
        if (sid >= ai->kf_count) {
            memset(out_bytes[k], 0, GRID_SIZE);
            continue;
        }
        const SpatialGrid* g = &ai->keyframes[sid].grid;
        for (uint32_t y = 0; y < GRID_SIZE; y++) {
            uint32_t best_x = 0;
            uint16_t best_a = 0;
            for (uint32_t x = 0; x < GRID_SIZE; x++) {
                uint16_t a = g->A[y * GRID_SIZE + x];
                if (a > best_a) { best_a = a; best_x = x; }
            }
            out_bytes[k][y] = (uint8_t)best_x;
        }
    }
}

/* Validate `span_bytes[0..len-1]` as a UTF-8 sequence of length `len`
 * and, if valid, return the summed agg_score_byte_full over the span.
 * Returns -1.0 on mismatch (lead byte's UTF-8 width != len, or a
 * continuation byte is missing) so the caller can skip this source. */
static double refine_span_score(
        const AggTables* tbl, const InputSignature* sig,
        const SpatialGrid* cand,
        const uint8_t* span_bytes, uint32_t y, int len) {
    if (len <= 0) return -1.0;
    if (utf8_lead_len(span_bytes[0]) != len) return -1.0;
    for (int i = 1; i < len; i++) {
        if (!utf8_is_cont(span_bytes[i])) return -1.0;
    }

    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        double in_R, in_G, in_B;
        input_signature_get(sig, y + (uint32_t)i, &in_R, &in_G, &in_B);
        double s = agg_score_byte_full(tbl, y + (uint32_t)i,
                                       span_bytes[i],
                                       in_R, in_G, in_B, cand);
        sum += s;
    }
    return sum;
}

/* Atomically replace rows [y, y+len) in `cand` with `new_x[0..len-1]`.
 * Clears every cell in the old span first so shorter replacements
 * never leave stray continuation bytes behind, then writes the new
 * cells with aggregated R/G/B and A clamped to the span's scores.
 * Updates cur_x[y..y+len-1] in place. */
static void refine_commit_span(
        SpatialGrid* cand, const AggTables* tbl,
        uint32_t y, int len,
        const uint8_t* new_x,
        const double* per_row_scores,
        uint32_t* cur_x) {
    /* Phase 1: clear the old span. */
    for (int i = 0; i < len; i++) {
        uint32_t old_x = cur_x[y + (uint32_t)i];
        if (old_x == UINT32_MAX) continue;
        uint32_t old = (y + (uint32_t)i) * GRID_SIZE + old_x;
        cand->A[old] = 0;
        cand->R[old] = 0;
        cand->G[old] = 0;
        cand->B[old] = 0;
    }
    /* Phase 2: write the new span. */
    for (int i = 0; i < len; i++) {
        uint32_t idx = (y + (uint32_t)i) * GRID_SIZE + (uint32_t)new_x[i];
        double s = per_row_scores ? per_row_scores[i] : 1.0;
        if (s > 65535.0) s = 65535.0;
        if (s < 1.0)     s = 1.0;
        cand->A[idx] = (uint16_t)s;
        cand->R[idx] = (uint8_t)tbl->R_mean[idx];
        cand->G[idx] = (uint8_t)tbl->G_mean[idx];
        cand->B[idx] = (uint8_t)tbl->B_mean[idx];
        cur_x[y + (uint32_t)i] = new_x[i];
    }
}

/* Cheap word-boundary predicate: returns 1 when the byte range
 * text[a..b-1] contains NO ASCII whitespace or punctuation, so the
 * two offsets lie in the same word. Works at the byte level because
 * UTF-8 continuation bytes (0x80..0xBF) are never whitespace nor
 * ASCII punctuation. */
static int refine_same_word(const char* text, uint32_t a, uint32_t b) {
    if (!text || b <= a) return 0;
    for (uint32_t i = a; i < b; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 0;
        if (c == '.' || c == ',' || c == '!' || c == '?') return 0;
        if (c == ':' || c == ';' || c == '(' || c == ')') return 0;
        if (c == '"' || c == '\'') return 0;
    }
    return 1;
}

/* ── ai_generate_refine — character-level + stickiness ────
 *
 * Revised pipeline (spec §3.1 / §3.4 plus the character-span fix):
 *
 *   1. encode input through the full layer pipeline
 *   2. spatial_match(MATCH_PREDICT) → top-k ids + scores (RGB-weighted
 *      cosine — discriminative even on tiny corpora where MATCH_GENERATE
 *      saturates)
 *   3. build focused AggTables from the top-k's NEXT-in-topic frames
 *   4. seed candidate from the top-1 next-in-topic grid (UTF-8 safe)
 *   5. character-span refinement: walk the seed in UTF-8 characters
 *      (1/2/3/4 rows) and for each span pick the winning source among
 *      {seed, top-k next frames}. A source is only considered when its
 *      bytes at this offset form a UTF-8 sequence of exactly the same
 *      length as the seed. Morpheme-word stickiness: the previous
 *      character's source KF earns a small bonus when the two chars
 *      lie inside the same word (no whitespace/punct between them).
 *   6. decode via grid_decode_text_utf8.
 *
 * Convergence: stop when characters-swapped drops below
 * GEN_REFINE_CONVERGE or after GEN_REFINE_ITERS iterations.
 */
uint32_t ai_generate_refine(SpatialAI* ai, const char* input_text,
                            char* out, uint32_t max_out,
                            float* out_match_similarity) {
    if (!ai || !input_text || !out || max_out == 0 || ai->kf_count == 0) {
        if (out && max_out > 0) out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 1. Encode input */
    SpatialGrid* in_grid = grid_create();
    if (!in_grid) {
        out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }
    layers_encode_clause(input_text, NULL, in_grid);
    update_rgb_directional(in_grid);
    apply_ema_to_grid(ai, in_grid);

    /* Zero-input short-circuit: empty or whitespace-only input produces
     * no grid activity, so retrieval is meaningless. Return empty. */
    if (grid_active_count(in_grid) == 0) {
        grid_destroy(in_grid);
        out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }

    /* 2. Top-k retrieval (MATCH_PREDICT — RGB-weighted cosine). */
    MatchContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.bucket_idx = &ai->bucket_idx;
    MatchResult r = spatial_match(ai, in_grid, MATCH_PREDICT, &ctx);

    if (r.topk_count == 0 || r.best_id >= ai->kf_count) {
        grid_destroy(in_grid);
        out[0] = '\0';
        if (out_match_similarity) *out_match_similarity = 0.0f;
        return 0;
    }
    if (out_match_similarity) *out_match_similarity = r.best_score;

    uint32_t ids[TOP_K];
    float    sc [TOP_K];
    uint32_t kcount = r.topk_count;
    for (uint32_t i = 0; i < kcount; i++) {
        ids[i] = r.topk[i].id;
        sc [i] = r.topk[i].score;
    }

    /* 3. Focused AggTables from top-k next-in-topic frames. */
    AggTables* tbl = agg_build_topk(ai, ids, sc, kcount, /*use_next_in_topic*/ 1);
    if (!tbl) {
        grid_destroy(in_grid);
        out[0] = '\0';
        return 0;
    }

    /* 4. Seed candidate with the top-1 next-in-topic grid (UTF-8 safe). */
    uint32_t seed_id = ai_next_in_topic(ai, ids[0]);
    if (seed_id >= ai->kf_count) seed_id = ids[0];

    SpatialGrid* cand = grid_create();
    if (!cand) {
        agg_destroy(tbl);
        grid_destroy(in_grid);
        out[0] = '\0';
        return 0;
    }
    grid_copy(cand, &ai->keyframes[seed_id].grid);

    /* Track current column per row so we can revert on swap. Derived
     * from seed via row-argmax (seed has exactly one active cell per
     * row in the common case; argmax handles accidental ties). */
    uint32_t cur_x[GRID_SIZE];
    for (uint32_t y = 0; y < GRID_SIZE; y++) {
        uint32_t best_x = UINT32_MAX;
        uint16_t best_a = 0;
        for (uint32_t x = 0; x < GRID_SIZE; x++) {
            uint32_t i = y * GRID_SIZE + x;
            if (cand->A[i] > best_a) { best_a = cand->A[i]; best_x = x; }
        }
        cur_x[y] = best_x;
    }

    /* 5. Pre-compute source byte streams (top-k next-in-topic) + seed
     *    decoded text for word-boundary checks. */
    uint32_t ids_next[TOP_K];
    for (uint32_t k = 0; k < kcount; k++) {
        ids_next[k] = ai_next_in_topic(ai, ids[k]);
    }

    uint8_t src_bytes[TOP_K][GRID_SIZE];
    memset(src_bytes, 0, sizeof(src_bytes));
    refine_collect_source_bytes(ai, ids_next, kcount, src_bytes);

    /* Build a byte_offset[y] → decoded_text_offset map for the seed.
     * grid_decode_text_utf8 reads one row per byte emitted, so the
     * cumulative byte count after decoding up to row y tells us where
     * in seed_text the character at row y begins. */
    char     seed_text[2048];
    uint32_t row_off[GRID_SIZE];        /* row y → byte offset in seed_text */
    for (uint32_t y = 0; y < GRID_SIZE; y++) row_off[y] = UINT32_MAX;
    {
        uint32_t total = grid_decode_text_utf8(cand, seed_text, sizeof(seed_text));
        /* Re-walk cand by character to populate row_off. */
        uint32_t y = 0, off = 0;
        while (y < GRID_SIZE && off < total) {
            if (cur_x[y] == UINT32_MAX) { y++; continue; }
            int len = utf8_lead_len((uint8_t)cur_x[y]);
            if (len <= 0) { y++; continue; }
            row_off[y] = off;
            off += (uint32_t)len;
            y   += (uint32_t)len;
        }
    }

    InputSignature sig;
    input_signature_compute(&sig, in_grid);

    /* 6. Character-span refinement loop. */
    int prev_changed = (int)GRID_SIZE + 1;
    for (int iter = 0; iter < GEN_REFINE_ITERS; iter++) {
        int     changed     = 0;
        int     prev_k      = -1;
        uint32_t prev_end   = UINT32_MAX;

        uint32_t y = 0;
        while (y < GRID_SIZE) {
            if (cur_x[y] == UINT32_MAX) { y++; continue; }
            uint8_t lead = (uint8_t)cur_x[y];
            int len = utf8_lead_len(lead);
            if (len <= 0) len = 1;
            if (y + (uint32_t)len > GRID_SIZE) break;

            /* Seed span as the baseline candidate. */
            uint8_t seed_span[4] = {0,0,0,0};
            for (int i = 0; i < len; i++) seed_span[i] = (uint8_t)cur_x[y + (uint32_t)i];

            double seed_score = refine_span_score(tbl, &sig, cand, seed_span, y, len);
            double best_score = seed_score;
            uint8_t best_span[4];
            memcpy(best_span, seed_span, (size_t)len);
            int best_k = -1;   /* -1 = keep seed */

            /* Evaluate each top-k source's span at this offset. */
            for (uint32_t k = 0; k < kcount; k++) {
                uint8_t span[4] = {0,0,0,0};
                for (int i = 0; i < len; i++) span[i] = src_bytes[k][y + (uint32_t)i];
                double s = refine_span_score(tbl, &sig, cand, span, y, len);
                if (s < 0.0) continue;

                /* Morpheme stickiness: reward sticking with the
                 * previous character's source when we haven't crossed
                 * a word boundary. Proportional to row activity so
                 * it scales with the AggTable's confidence. */
                if (prev_k >= 0 && (int)k == prev_k &&
                    prev_end != UINT32_MAX && row_off[y] != UINT32_MAX &&
                    refine_same_word(seed_text, prev_end, row_off[y])) {
                    s += GEN_STICKY_BONUS_FRAC * (double)len * tbl->row_total_A[y];
                }

                if (s > best_score) {
                    best_score = s;
                    memcpy(best_span, span, (size_t)len);
                    best_k = (int)k;
                }
            }

            /* Commit span if it differs from the seed. */
            int differs = 0;
            for (int i = 0; i < len; i++) {
                if (best_span[i] != seed_span[i]) { differs = 1; break; }
            }
            if (differs) {
                /* Per-row score distribution: equal share across span. */
                double per_row[4];
                double share = best_score / (double)len;
                if (share < 1.0) share = 1.0;
                for (int i = 0; i < len; i++) per_row[i] = share;
                refine_commit_span(cand, tbl, y, len, best_span, per_row, cur_x);
                changed++;
            }

            /* Update stickiness state. */
            prev_k  = best_k;
            prev_end = (row_off[y] != UINT32_MAX)
                       ? row_off[y] + (uint32_t)len
                       : UINT32_MAX;
            y += (uint32_t)len;
        }

        if (changed < GEN_REFINE_CONVERGE) break;
        if (changed >= prev_changed) break;
        prev_changed = changed;
    }

    /* 7. Decode. */
    uint32_t written = grid_decode_text_utf8(cand, out, max_out);

    grid_destroy(cand);
    grid_destroy(in_grid);
    agg_destroy(tbl);
    return written;
}
