# Anki CSV Builder

A Streamlit web application that generates Dutch language learning flashcards for Anki using OpenAI's API. Creates high-quality cloze deletion cards with translations, definitions, and collocations tailored to specific CEFR language levels.

## Features

### 🎯 CEFR-Level Adaptation
- Supports A1, A2, B1, B2, C1, C2 proficiency levels
- Automatically adjusts vocabulary complexity and explanations
- Includes signal words (signaalwoorden) for B1+ levels (50% probability)

### 🃏 Smart Card Generation
- **Cloze deletion format** with proper Anki syntax (`{{c1::word}}`)
- **Separable verb support** with multi-cloze markers (`{{c1::stem}} ... {{c2::particle}}`)
- **Three collocations** per card with contextual usage
- **Bilingual definitions** (Dutch + L1 language)
- **Pronunciation guides** and grammatical information

### 🌍 Multi-Language Support
- **L1 Languages**: English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- Dynamic UI language switching
- Culturally appropriate field names and instructions

### 📊 Flexible Input/Output
- **Input formats**: Plain text, TSV, Markdown tables, em-dash separated
- **Export options**: CSV (pipe-delimited), Anki .apkg packages
- **Batch processing** with progress tracking

### 🤖 Advanced AI Integration
- **OpenAI Responses API** integration (not Chat API)
- **Multiple model support**: GPT-4o, GPT-4o-mini, o1-preview, o3-mini
- **Automatic retry logic** for unsupported temperature settings
- **JSON extraction** with regex fallback for wrapped responses

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/anki-csv-builder.git
   cd anki-csv-builder
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Get your API key from [OpenAI](https://platform.openai.com/api-keys)
   - Enter it in the app's sidebar when running

## Usage

### Starting the Application

```bash
streamlit run anki_csv_builder.py
```

The app will open in your browser at `http://localhost:8501`.

### Basic Workflow

1. **Configure settings** in the sidebar:
   - Enter your OpenAI API key
   - Select your L1 language
   - Choose CEFR level
   - Pick OpenAI model

2. **Input Dutch words**:
   - One word per line, or
   - TSV format: `dutch_word    translation`, or
   - Markdown table format, or
   - Em-dash format: `dutch_word — translation`

3. **Generate cards**:
   - Click "Generate Anki Cards"
   - Monitor progress in the progress bar
   - Review generated cards in the preview

4. **Export**:
   - Download CSV file for manual Anki import, or
   - Download .apkg file for direct Anki import

### Input Examples

**Plain text**:
```
huis
auto
boek
```

**TSV format**:
```
huis    house
auto    car
boek    book
```

**Markdown table**:
```
| Dutch | English |
|-------|---------|
| huis  | house   |
| auto  | car     |
```

## Project Architecture

### Configuration-First Design
- `config.py`: All configurable constants, templates, and demo data
- Comprehensive fallbacks for robust operation
- Easy customization without touching core logic

### Prompt Engineering Pipeline
- `prompts.py`: CEFR-level specific instruction generation
- Dynamic L1 language support
- Deterministic signal word inclusion based on proficiency level

### Data Processing Flow
1. **Input parsing** → Multiple format support
2. **OpenAI API call** → JSON response extraction
3. **Sanitization** → Cloze marker fixing, pipe character handling
4. **Validation** → Business rule enforcement
5. **Export** → CSV/APKG generation

### Key Functions

#### `sanitize()`
Handles cloze marker escaping and special character replacement:
- Converts single braces to double braces for Anki compatibility
- Replaces pipe characters to prevent CSV corruption

#### `validate_card()`
Enforces quality standards:
- Exactly 3 collocations required
- Maximum 2 words for L1 translations
- Proper cloze marker presence validation

## Anki Template System

The app includes responsive HTML templates optimized for both desktop and mobile Anki clients:

- **Front template**: Shows cloze with contextual information
- **Back template**: Reveals answer with full details
- **Styling**: CSS custom properties for responsive design
- **JavaScript**: Dynamic collocation rendering with numbering

## Development

### File Structure
```
anki-csv-builder/
├── anki_csv_builder.py    # Main Streamlit app
├── config.py              # Configuration and constants
├── prompts.py             # CEFR-aware prompt generation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Debug Mode
Enable debug mode in the sidebar to:
- View raw OpenAI API responses
- Inspect JSON extraction process
- Monitor sanitization steps
- Track validation results

### Testing Separable Verbs
Test with Dutch separable verbs like "opruimen" (to clean up) to ensure proper multi-cloze handling:
- Input: `opruimen`
- Expected output: `{{c1::ruim}} ... {{c2::op}}`

## Configuration

### OpenAI Models
Supported models (text-generation only):
- `gpt-4o` (recommended)
- `gpt-4o-mini` (faster, cheaper)
- `o1-preview` (advanced reasoning)
- `o3-mini` (latest model)

### CEFR Levels
- **A1-A2**: Basic vocabulary, simple explanations
- **B1-B2**: Intermediate complexity, signal words included
- **C1-C2**: Advanced vocabulary, nuanced explanations

## Troubleshooting

### Common Issues

1. **"No JSON found in response"**:
   - Enable debug mode to inspect raw response
   - Try a different model
   - Check API key validity

2. **Cloze markers not working**:
   - Ensure double curly braces: `{{c1::word}}`
   - Check sanitization output in debug mode

3. **Cards failing validation**:
   - Review collocation count (must be exactly 3)
   - Check L1 translation length (max 2 words)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code patterns
4. Test with various input formats
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and feature requests, please use the GitHub issue tracker.

