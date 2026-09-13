import { getShouldProcessPrompt } from 'features/dynamicPrompts/util/getShouldProcessPrompt';
import { describe, expect, it } from 'vitest';

describe('getShouldProcessPrompt', () => {
  it('detects variant syntax', () => {
    expect(getShouldProcessPrompt('a {red|blue} car')).toBe(true);
  });

  it('detects a wildcard reference', () => {
    expect(getShouldProcessPrompt('a woman, __poses__')).toBe(true);
  });

  it('detects a wildcard in a subfolder', () => {
    expect(getShouldProcessPrompt('a woman, __dgirls/Pose_Yoga__')).toBe(true);
  });

  it('leaves a plain prompt alone', () => {
    expect(getShouldProcessPrompt('a woman standing in a field')).toBe(false);
  });

  it('does not treat a lone underscore pair as a wildcard', () => {
    expect(getShouldProcessPrompt('a woman__ standing')).toBe(false);
  });
});
