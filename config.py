"""
Конфигурация для Anki CSV Builder
"""

from typing import List, Dict, Tuple

# ==========================
# Модели: дефолтный список + динамическая подгрузка из API
# ==========================

DEFAULT_MODELS: List[str] = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
]

_PREFERRED_ORDER: Dict[str, int] = {  # чем меньше число — тем выше в списке
    "gpt-5": 0,
    "gpt-5-mini": 1,
    "gpt-5-nano": 2,
    "gpt-4.1": 3,
    "gpt-4o": 4,
    "gpt-4o-mini": 5,
    "o3": 6,
    "o3-mini": 7,
}

# Модели, которые исключаем по подстроке в ID (нам нужен именно текст-генератор)
_BLOCK_SUBSTRINGS: Tuple[str, ...] = (
    "audio", "realtime",           # gpt-4o-audio-*, gpt-4o-realtime-*
    "embed", "embedding",          # text-embedding-*
    "whisper", "asr", "transcribe", "speech", "tts",  # ASR/TTS
    "moderation",                  # модерация
    "search",                      # поисковые/вспомогательные
    "vision", "vision-preview",    # чисто визуальные/превью
    "distill", "distilled",        # дистиллированные спец-модели
    "batch", "preview"             # служебные/превью/батчевые
)

# Разрешённые семейства (по префиксу) для текстовой генерации
_ALLOWED_PREFIXES: Tuple[str, ...] = ("gpt-5", "gpt-4.1", "gpt-4o", "o3")

# ==========================
# Системный промпт
# ==========================

PROMPT_SYSTEM: str = (
    "Ты — опытный лексикограф NL→RU и автор учебных материалов. "
    "Сгенерируй СТРОГО JSON-объект карточки Anki со структурой: "
    "{woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short}.\n"
    "ОБЩИЕ ПРАВИЛА (ОЧЕНЬ ВАЖНО):\n"
    "• Верни ТОЛЬКО JSON БЕЗ пояснений и форматирования.\n"
    "• НИ ОДНО поле не пустое. Запрещены пустые строки.\n"
    "• Символ '|' в текстах запрещён.\n"
    "• Если дано def_nl — строго следуй ему; не меняй базовое значение слова.\n"
    "• Сохраняй часть речи: ru_short должен соответствовать части речи слова "
    "(глагол→инфинитив; существительное→существительное; прилагательное→прилагательное).\n"
    "• ru_sentence — ТОЧНЫЙ перевод NL-предложения, без перефраза.\n"
    "• cloze_sentence — одно короткое естественное NL-предложение (8–14 слов, настоящее время, "
    "без имён/цифр/кавычек); целевое слово внутри {{c1::…}}.\n"
    "  Если слово — разделимый глагол: {{c1::stam}} … {{c2::partikel}}. Иначе только {{c1::…}}.\n"
    "• collocaties — РОВНО 3 частотные связки, разделитель '; ' (точка с запятой и пробел).\n"
    "  Каждая связка — 2–3 слова с целевым словом в естественной форме. Нельзя: бессмысленные пары "
    "(например, 'een grote caissière'), редкие/книжные, имена собственные.\n"
    "• Избегай редкой лексики; используй A2–B1 вокруг целевого слова.\n\n"
    "ФОРМАТ ВЫВОДА: один JSON-объект с ключами: woord, cloze_sentence, ru_sentence, collocaties, def_nl, ru_short.\n\n"
    "ПРИМЕРЫ (стиль, НЕ копируй слова):\n"
    "// Существительное\n"
    "{\"woord\": \"boodschap\", \"cloze_sentence\": \"Hij doet elke dag de {{c1::boodschap}}.\", "
    "\"ru_sentence\": \"Он делает покупки каждый день.\", "
    "\"collocaties\": \"boodschappen doen; een boodschap doorgeven; een duidelijke boodschap\", "
    "\"def_nl\": \"iets wat je wilt zeggen of inkopen die je doet\", \"ru_short\": \"покупка; послание\"}\n"
    "// Разделимый глагол\n"
    "{\"woord\": \"opruimen\", \"cloze_sentence\": \"Na het eten {{c1::ruimt}} hij de tafel {{c2::op}}.\", "
    "\"ru_sentence\": \"После еды он убирает со стола.\", "
    "\"collocaties\": \"de kamer opruimen; speelgoed opruimen; netjes opruimen\", "
    "\"def_nl\": \"iets op zijn plaats leggen zodat het netjes is\", \"ru_short\": \"убирать\"}\n"
    "// Прилагательное\n"
    "{\"woord\": \"streng\", \"cloze_sentence\": \"De docent is vandaag {{c1::streng}}.\", "
    "\"ru_sentence\": \"Преподаватель сегодня строгий.\", "
    "\"collocaties\": \"strenge regels; een strenge docent; streng optreden\", "
    "\"def_nl\": \"met veel eisen en weinig toelating\", \"ru_short\": \"строгий\"}"
)

# ==========================
# Демо-данные
# ==========================

DEMO_WORDS: List[Dict[str, str]] = [
    {"woord": "aanraken", "def_nl": "iets met je hand of een ander deel van je lichaam voelen"},
    {"woord": "begrijpen", "def_nl": "snappen wat iets betekent of inhoudt"},
    {"woord": "gillen", "def_nl": "hard en hoog schreeuwen"},
    {"woord": "kloppen", "def_nl": "met regelmaat bonzen of tikken"},
    {"woord": "toestaan", "def_nl": "goedkeuren of laten gebeuren"},
    {"woord": "opruimen", "def_nl": "iets netjes maken door het op zijn plaats te leggen"},
]

# ==========================
# Настройки UI
# ==========================

PAGE_TITLE: str = "Anki CSV Builder"
PAGE_LAYOUT: str = "wide"

# Настройки temperature slider
TEMPERATURE_MIN: float = 0.2
TEMPERATURE_MAX: float = 0.8
TEMPERATURE_DEFAULT: float = 0.4
TEMPERATURE_STEP: float = 0.1

# Настройки CSV
CSV_DELIMITER: str = '|'
CSV_LINETERMINATOR: str = '\n'

# Настройки предпросмотра
PREVIEW_LIMIT: int = 20

# Задержка между API запросами (в секундах)
API_REQUEST_DELAY: float = 0.1

# ==========================
# Заголовки CSV
# ==========================

CSV_HEADERS: List[str] = [
    "NL-слово",
    "Предложение NL (с cloze)",
    "Перевод RU",
    "Коллокации",
    "Определение NL",
    "Перевод слова RU",
]

# ==========================
# Сообщения и подсказки
# ==========================

MESSAGES = {
    "demo_loaded": "🔁 Используется демо-набор из 6 слов",
    "no_api_key": "Укажи OPENAI_API_KEY в Secrets или в поле слева.",
    "temperature_unavailable": "Температура недоступна для этой модели; она будет проигнорирована.",
    "help_temperature": "Лучшее качество — gpt-5 (если доступен); баланс — gpt-4.1; быстрее/дешевле — gpt-4o / gpt-5-mini.",
    "help_stream": "Streaming в Responses API: финальный JSON будет доступен после завершения стрима",
    "placeholder_custom_model": "например, gpt-5-2025-08-07",
    "footer_tips": (
        "Советы: 1) добавляй качественные NL-дефиниции во вход — это улучшит примеры; "
        "2) если видишь странные коллокации — перезапусти генерацию для конкретного слова, "
        "3) символ '|' в текстах заменяется на '∣'."
    )
}

# ==========================
# Функции для работы с конфигурацией
# ==========================

def get_preferred_order() -> Dict[str, int]:
    """Возвращает словарь предпочтительного порядка моделей"""
    return _PREFERRED_ORDER.copy()

def get_block_substrings() -> Tuple[str, ...]:
    """Возвращает кортеж запрещенных подстрок"""
    return _BLOCK_SUBSTRINGS

def get_allowed_prefixes() -> Tuple[str, ...]:
    """Возвращает кортеж разрешенных префиксов"""
    return _ALLOWED_PREFIXES
