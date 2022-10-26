import { createSlice } from '@reduxjs/toolkit'
import type { PayloadAction } from '@reduxjs/toolkit'

// export interface CounterState {
//   value: number
// }

// data["browser"] = {
//   "suggestions": suggestions_children,
//   "tests": children,
//   "user": self.user,
//   "topic": self.current_topic,
//   "topic_description": self.test_tree.loc[topic_marker_id]["description"] if topic_marker_id is not None else "",
//   "topic_marker_id": topic_marker_id if topic_marker_id is not None else uuid.uuid4().hex,
//   "score_filter": score_filter,
//   "disable_suggestions": False,
//   "read_only": False,
//   "score_columns": self.score_columns,
//   "suggestions_error": self._suggestions_error,
//   "generator_options": [str(x) for x in self.generators.keys()] if isinstance(self.generators, dict) else [self.active_generator],
//   "active_generator": self.active_generator,
//   "mode": self.mode,
//   "mode_options": self.mode_options,
//   "test_tree_name": self.test_tree.name
// }

// export interface TestTreeState {
//   // topic: string;
//   // suggestions: any[];
//   // tests: any[];
//   selections: any;
//   // user: string;
//   loading_suggestions: boolean;
//   max_suggestions: number;
//   suggestions_pos: number;
//   suggestionsDropHighlighted: number;
//   // score_filter: number;
//   do_score_filter: boolean;
//   // filter_text: string;
//   experiment_pos: number;
//   timerExpired: boolean;
//   experiment_locations: any[];
//   experiment: boolean;
//   value2Filter: string;
//   test_types?: any[];
//   test_type_parts?: any[];
//   // score_columns?: any[];
//   // test_tree_name?: any;
//   // topic_description?: string;
//   // read_only?: boolean;
//   topicFilter?: string;
//   value1Filter?: string;
//   comparatorFilter?: string;
//   // disable_suggestions?: boolean;
//   // mode_options?: any[];
//   // generator_options?: any[];
//   // active_generator?: string;
//   // mode?: string;
//   // suggestions_error?: string;
//   // topic_marker_id?: string;
// }

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

    updateFilterText: (state, action: PayloadAction<string>) => {
      state.filter_text = action.payload;
    },

    updateSuggestions: (state, action: PayloadAction<any[]>) => {
      state.suggestions = [...action.payload];
    }
  },
})

// Action creators are generated for each case reducer function
export const { refresh, updateGenerator, updateTopicDescription, updateFilterText, updateSuggestions } = testTreeSlice.actions

export default testTreeSlice.reducer