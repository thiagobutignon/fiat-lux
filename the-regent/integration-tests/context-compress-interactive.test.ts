/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { expect, describe, it, beforeEach, afterEach } from 'vitest';
import { TestRig, type } from './test-helper.js';

describe('Interactive Mode', () => {
  let rig: TestRig;

  beforeEach(() => {
    rig = new TestRig();
  });

  afterEach(async () => {
    await rig.cleanup();
  });

  it.skipIf(process.platform === 'win32')(
    'should trigger chat compression with /compress command',
    async () => {
      await rig.setup('interactive-compress-test');

      const { ptyProcess } = rig.runInteractive();
      await rig.ensureReadyForInput(ptyProcess);

      const longPrompt =
        'Dont do anything except returning a 1000 token long paragragh with the <name of the scientist who discovered theory of relativity> at the end to indicate end of response. This is a moderately long sentence.';

      await type(ptyProcess, longPrompt);
      await type(ptyProcess, '\r');

      await rig.waitForText('einstein', 25000);

      await type(ptyProcess, '/compress');
      // A small delay to allow React to re-render the command list.
      await new Promise((resolve) => setTimeout(resolve, 100));
      await type(ptyProcess, '\r');

      const foundEvent = await rig.waitForTelemetryEvent(
        'chat_compression',
        90000,
      );
      expect(foundEvent, 'chat_compression telemetry event was not found').toBe(
        true,
      );
    },
  );

  it.skipIf(process.platform === 'win32')(
    'should handle compression failure on token inflation',
    async () => {
      await rig.setup('interactive-compress-test');

      const { ptyProcess } = rig.runInteractive();
      await rig.ensureReadyForInput(ptyProcess);

      await type(ptyProcess, '/compress');
      await new Promise((resolve) => setTimeout(resolve, 100));
      await type(ptyProcess, '\r');

      const compressionFailed = await rig.waitForText(
        'compression was not beneficial',
        25000,
      );

      expect(compressionFailed).toBe(true);
    },
  );
});
