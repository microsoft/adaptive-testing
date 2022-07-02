
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
