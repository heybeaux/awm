/**
 * AWM CLI — NDJSON bridge between the Python benchmark and the
 * TypeScript AWM core + equity-store.
 *
 * Protocol:
 *   stdin:  one JSON object per line
 *   stdout: one JSON object per line (response to the corresponding input)
 *
 * Commands:
 *   {"action":"predict","ticker":"AAPL","regime":"trending_up","trace_id":"..."}
 *     → {"trace_id":"...","p_up":0.62,"confidence":0.73,"arm":"trending_up",
 *        "alpha":3.5,"beta":2.0,"observations":4}
 *
 *   {"action":"record","ticker":"AAPL","regime":"trending_up","outcome":1,
 *    "trace_id":"..."}
 *     → {"trace_id":"...","ok":true}
 *
 *   {"action":"reset"}
 *     → {"ok":true}
 *
 *   {"action":"shutdown"}
 *     → {"ok":true} (process exits)
 *
 * The belief state maps:
 *   stepType    = regime (trending_up / trending_down / mean_reverting / quiet)
 *   profileSlug = ticker (AAPL, SPY, ...)
 *
 * Beliefs accumulate per-(ticker, regime) via Beta(alpha,beta). Posterior
 * mean is P(up); confidence is 1 − var/max_var.
 */

import * as readline from 'node:readline';
import { BeliefEngine } from '@heybeaux/awm-core';
import { EquityStore, tickerToProfile } from '@heybeaux/awm-equity-store';

interface PredictMsg {
  action: 'predict';
  ticker: string;
  regime: string;
  trace_id?: string;
}

interface RecordMsg {
  action: 'record';
  ticker: string;
  regime: string;
  outcome: 0 | 1 | boolean;
  trace_id?: string;
}

interface ResetMsg {
  action: 'reset';
}

interface ShutdownMsg {
  action: 'shutdown';
}

type Msg = PredictMsg | RecordMsg | ResetMsg | ShutdownMsg;

function writeLine(obj: unknown): void {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

async function main(): Promise<void> {
  const dbPath = process.env.AWM_DB_PATH ?? ':memory:';
  let store = new EquityStore({ dbPath });
  let engine = new BeliefEngine(store);

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const rawLine of rl) {
    const line = rawLine.trim();
    if (!line) continue;

    let msg: Msg;
    try {
      msg = JSON.parse(line) as Msg;
    } catch (err) {
      writeLine({ ok: false, error: `parse: ${(err as Error).message}` });
      continue;
    }

    try {
      if (msg.action === 'predict') {
        const profile = tickerToProfile(msg.ticker);
        const belief = await engine.getBelief(msg.regime, profile);
        const p_up = engine.successProbability(belief);
        const confidence = engine.confidence(belief);
        writeLine({
          trace_id: msg.trace_id ?? null,
          p_up,
          confidence,
          arm: msg.regime,
          alpha: belief.alpha,
          beta: belief.beta,
          observations: belief.observations,
        });
      } else if (msg.action === 'record') {
        const profile = tickerToProfile(msg.ticker);
        const passed =
          typeof msg.outcome === 'boolean' ? msg.outcome : Number(msg.outcome) === 1;
        await engine.update(msg.regime, profile, passed);
        writeLine({ trace_id: msg.trace_id ?? null, ok: true });
      } else if (msg.action === 'reset') {
        store.reset();
        engine = new BeliefEngine(store);
        writeLine({ ok: true });
      } else if (msg.action === 'shutdown') {
        writeLine({ ok: true });
        store.close();
        process.exit(0);
      } else {
        writeLine({ ok: false, error: `unknown action: ${(msg as { action?: string }).action}` });
      }
    } catch (err) {
      writeLine({
        ok: false,
        error: `handler: ${(err as Error).message}`,
        stack: (err as Error).stack,
      });
    }
  }

  store.close();
}

main().catch((err) => {
  process.stderr.write(`[awm_cli] fatal: ${(err as Error).stack ?? err}\n`);
  process.exit(1);
});
