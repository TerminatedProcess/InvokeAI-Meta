import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  combinatorialChanged,
  selectDynamicPromptsCombinatorial,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsCombinatorial = () => {
  const combinatorial = useAppSelector(selectDynamicPromptsCombinatorial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(combinatorialChanged(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl w="auto" flexShrink={0}>
      <FormLabel>{t('dynamicPrompts.combinatorial')}</FormLabel>
      <Switch isChecked={combinatorial} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsCombinatorial);
