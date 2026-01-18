class Workflow:
    def __init__(self):
        logger.info("Avvio Orchestrator...")
        self.classifier = IntentClassifierAgent()
        self.finder = AudioFinderAgent()
        self.designer = SoundDesignerAgent()
        self.analyst = AudioAnalystAgent()
        self.writer = HumanizerAgent()
        logger.info("Sistema pronto.")