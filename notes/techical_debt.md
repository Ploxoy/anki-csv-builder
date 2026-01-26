Title: Function-Calling & Structured Outputs strategy — 🚧 In progress

Summary:
Схема Response API и fallback на text-parsing уже обёрнуты в `core/llm_clients`,
но сама поддержка function-calling пока не внедрена. Требуется спроектировать
ограниченный диспетчер инструментов и безопасный тулсет для генерации карточек.

Status / Notes:
- `app.ui_helpers.probe_response_format_support` выполняет пробный вызов и теперь
  кеширует результат в `cache/response_format.json` (поддержка по модели + причина).
- `core/generation.generate_card` всегда работает через `send_responses_request`
  и fallback на текст; инструментов и `response.output_tool_calls` в проекте пока нет.
- При отсутствии schema модель возвращает текст, JSON парсим локально (см.
  `extract_json_block`), validation/repair выполняется без tools.
- Тестов на обёртку Responses API и будущий диспетчер инструментов нет;
  текущее покрытие ограничивается генерацией/экспортом.
- Run report 2.0 показывает долю schema-fallback и токены, но метрики по tool-calls
  пока отсутствуют.

Next steps:
- Определить минимальный список read-only инструментов (`check_separable`,
  `get_cached_collocations`, другие проверки) и их контракт.
- Реализовать диспетчер function-calling: маршрутизация запросов, лимиты,
  логирование; обеспечить совместимость со схемой Responses.
- Добавить unit-тесты: probes, обработка отказа от schema, happy-path tool call.
- Расширить Run report и метрики, чтобы отслеживать долю tool-вызовов,
  fallbacks и стоимость.
- Добавить тесты на пробу schema (успех/провал) и обновление кеша при смене модели.
