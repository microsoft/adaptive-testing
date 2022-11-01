import { createSlice } from '@reduxjs/toolkit'
import type { PayloadAction } from '@reduxjs/toolkit'

export interface TestTreeState {
  topic: string;
  suggestions: Object;
  tests: Object;
  user: string;
  score_filter: number;
  filter_text: string;
  score_columns?: any[];
  test_tree_name?: any;
  topic_description?: string;
  read_only?: boolean;
  disable_suggestions?: boolean;
  mode_options?: any[];
  generator_options?: any[];
  active_generator?: string;
  mode?: string;
  suggestions_error?: string;
  topic_marker_id?: string;
}

const initialState: TestTreeState = {
  topic: "/",
  suggestions: [],
  tests: [],
  user: "anonymous",
  score_filter: 0.3,
  filter_text: "",
}

// See https://redux-toolkit.js.org/usage/immer-reducers#immer-usage-patterns for explanation
// of how to use Immer to write reducers
export const testTreeSlice = createSlice({
  name: 'testTree',
  initialState,
  reducers: {
    refresh: (state, action: PayloadAction<TestTreeState>) => {
      return action.payload;
    },

    updateGenerator: (state, action: PayloadAction<string>) => {
      state.active_generator = action.payload;
    },

    updateTopicDescription: (state, action: PayloadAction<string>) => {
      state.topic_description = action.payload;
    },

    updateSuggestions: (state, action: PayloadAction<any[]>) => {
      state.suggestions = [...action.payload];
    }
  },
})

// Action creators are generated for each case reducer function
export const { refresh, updateGenerator, updateTopicDescription, updateSuggestions } = testTreeSlice.actions

export default testTreeSlice.reducer