
// The base class for all communication data over websocket
export class CommData {
  // Using snake case because these objects will be parsed in Python backend
  readonly event_id: string;
  data: object;
  constructor(event_id: string, data: object) {
    this.event_id = event_id;
    this.data = data;
  }
}


export class BrowserEvent extends CommData {
  constructor(action: BrowserAction) {
    super("browser", action);
  }
}


export class BrowserAction {
  action: string
  constructor(action: string) {
    this.action = action;
  }
}


export class FinishTopicDescriptionAction extends BrowserAction {
  topic_marker_id: string;
  description: string;

  constructor(topic_marker_id: string, description: string) {
    super("change_description");
    this.topic_marker_id = topic_marker_id;
    this.description = description;
  }
}


export class RedrawAction extends BrowserAction {
  constructor() {
    super("redraw");
  }
}


export class GenerateSuggestionsAction extends BrowserAction {
  data: object;

  constructor(data: object) {
    super("generate_suggestions");
    this.data = data;
  }
}


export class ClearSuggestionsAction extends BrowserAction {
  constructor() {
    super("clear_suggestions");
  }
}


export class SetFirstModelAction extends BrowserAction {
  model: string
  constructor(model: string) {
    super("set_first_model");
    this.model = model;
  }
}


export class ChangeGeneratorAction extends BrowserAction {
  generator: string
  constructor(generator: string) {
    super("active_generator");
    this.generator = generator;
  }
}


export class ChangeModeAction extends BrowserAction {
  mode: string
  constructor(mode: string) {
    super("mode");
    this.mode = mode;
  }
}


export class AddTopicAction extends BrowserAction {
  constructor() {
    super("add_new_topic");
  }
}


export class AddTestAction extends BrowserAction {
  constructor() {
    super("add_new_test");
  }
}


export class ChangeFilterAction extends BrowserAction {
  filter_text: string
  constructor(filter_text: string) {
    super("change_filter");
    this.filter_text = filter_text;
  }
}

export class ChangeTopicAction extends BrowserAction {
  topic: string
  constructor(topic: string) {
    super("change_topic");
    this.topic = topic;
  }
}




