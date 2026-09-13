import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { maxPromptsChanged, selectDynamicPromptsMaxPrompts } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 100,
  sliderMin: 1,
  sliderMax: 1000,
  numberInputMin: 1,
  numberInputMax: 10000,
  fineStep: 1,
  coarseStep: 10,
};

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector(selectDynamicPromptsMaxPrompts);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(maxPromptsChanged(v));
    },
    [dispatch]
  );

  // Caps the prompt count in both modes: combinations enumerated, or random draws taken.
  return (
    <FormControl>
      <InformationalPopover feature="dynamicPromptsMaxPrompts" inPortal={false}>
        <FormLabel>{t('dynamicPrompts.maxPrompts')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        value={maxPrompts}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        value={maxPrompts}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
