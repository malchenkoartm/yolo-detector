from interfaces import ICommandParser, Command, CommandType

class CommandParser(ICommandParser):
    __ZOOM_IN  = ["приблизить", "приблизь", "увеличить", "вперед", "вперёд", "перед", "перёд"]
    __ZOOM_OUT = ["отдалить", "отдали", "уменьшить", "удалить", "удали", "назад", "зад"]
    __EXIT     = ["выйти", "выход", "стоп"]
    __PLACE    = ["найти", "найди", "поиск", "искать", "ищи"]
    __ADD      = ["добавить", "добавь", "плюс"]
    __CONF     = ["уверенность", "уверен", "порог"]
    __FOLLOW   = ["следить", "следи", "след"]
    
    @classmethod
    def get_triggers(cls):
        return cls.__ZOOM_IN + cls.__ZOOM_OUT + cls.__EXIT + cls.__PLACE + cls.__ADD + cls.__CONF + cls.__FOLLOW

    def parse(self, text: str, to_add: bool = False) -> Command:
        text = text.strip().lower()

        if any(w in text for w in self.__ZOOM_IN):
            return Command(CommandType.ZOOM_IN)
        if any(w in text for w in self.__ZOOM_OUT):
            return Command(CommandType.ZOOM_OUT)
        if any(w in text for w in self.__EXIT):
            return Command(CommandType.EXIT)

        for w in self.__PLACE:
            if text.startswith(w):
                return Command(CommandType.PLACE, text=text[len(w):].strip())

        for w in self.__ADD:
            if text.startswith(w):
                return Command(CommandType.ADD, text=text[len(w):].strip())
            
        for w in self.__FOLLOW:
            if text.startswith(w):
                return Command(CommandType.FOLLOW, text=text[len(w):].strip())

        for w in self.__CONF:
            if text.startswith(w):
                return Command(CommandType.CONF, text=text[len(w):].strip())
            
        if text:
            return Command(CommandType.ADD if to_add else CommandType.PLACE, text=text)

        return Command(CommandType.UNKNOWN)