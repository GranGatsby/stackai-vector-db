# StackAI Vector Database

Una API REST para indexar y consultar documentos en una base de datos vectorial, desarrollada como parte del proceso de entrevista técnica de StackAI.

## 🚀 Características

- **API REST completa** para operaciones CRUD en bibliotecas, documentos y chunks
- **Búsqueda vectorial k-NN** con múltiples algoritmos de indexación
- **Arquitectura limpia** siguiendo principios DDD y SOLID
- **Tipado estático** completo con MyPy
- **Containerización** con Docker
- **Herramientas de desarrollo** integradas (Black, Ruff, Pre-commit)

## 📋 Requisitos

- Python 3.12+
- Docker (opcional)
- Clave API de Cohere para embeddings

## 🛠️ Instalación

### Desarrollo Local

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd stackai-vector-db
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate  # Windows
   ```

3. **Instalar dependencias de desarrollo**
   ```bash
   make setup
   # o manualmente:
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Configurar variables de entorno**
   ```bash
   cp env.example .env
   # Editar .env con tu clave API de Cohere
   ```

### Docker

```bash
# Construir imagen
make docker-build

# Ejecutar contenedor
make docker-run
```

## 🏃‍♂️ Uso

### Ejecutar la aplicación

```bash
# Desarrollo
make run

# O directamente
python -m app.main
```

La API estará disponible en `http://localhost:8000`

### Documentación de la API

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## 🧪 Testing

```bash
# Ejecutar todos los tests
make test

# Con cobertura
pytest tests/ --cov=app --cov-report=html
```

## 🔧 Herramientas de Desarrollo

```bash
# Formatear código
make format

# Linting
make lint

# Type checking
make type-check

# Ejecutar pre-commit hooks
make pre-commit

# Todas las verificaciones
make check
```

## 📁 Estructura del Proyecto

```
stackai-vector-db/
├── app/
│   ├── api/v1/routers/     # Endpoints de la API
│   ├── core/               # Configuración y logging
│   ├── domain/             # Entidades y lógica de negocio
│   ├── services/           # Casos de uso de la aplicación
│   ├── clients/            # Clientes externos (Cohere)
│   ├── repositories/       # Interfaces y adaptadores de datos
│   ├── indexes/            # Algoritmos de indexación vectorial
│   ├── schemas/            # DTOs de Pydantic
│   ├── utils/              # Utilidades
│   └── main.py             # Punto de entrada de la aplicación
├── tests/                  # Tests unitarios e integración
├── Dockerfile              # Imagen de Docker
├── Makefile               # Comandos de desarrollo
├── pyproject.toml         # Configuración del proyecto
└── README.md              # Este archivo
```

## 🏗️ Arquitectura

El proyecto sigue una arquitectura limpia basada en Domain-Driven Design (DDD):

- **API Layer**: Routers de FastAPI que manejan HTTP
- **Service Layer**: Lógica de aplicación y casos de uso
- **Domain Layer**: Entidades y reglas de negocio
- **Repository Layer**: Abstracción de persistencia
- **Infrastructure**: Implementaciones concretas (índices, clientes)

## ⚙️ Configuración

Todas las configuraciones se manejan a través de variables de entorno usando Pydantic Settings:

- `COHERE_API_KEY`: Clave API de Cohere (requerida)
- `DEFAULT_INDEX_TYPE`: Tipo de índice por defecto (linear, kdtree, ivf)
- `MAX_CHUNKS_PER_LIBRARY`: Máximo número de chunks por biblioteca
- Ver `env.example` para todas las opciones

## 🔍 Algoritmos de Indexación

### Implementados

1. **Linear Scan**: O(N) búsqueda, O(N) espacio
2. **KD-Tree**: ~O(N log N) construcción, eficiente en bajas dimensiones
3. **IVF-like**: Cuantización gruesa con listas invertidas

### Consideraciones de Concurrencia

- **Single-writer principle**: Operaciones de escritura protegidas con locks
- **Read/Write locks**: Lecturas concurrentes permitidas
- **Copy-on-write**: Lecturas desde snapshots inmutables durante reconstrucción

## 🐛 Troubleshooting

### Errores Comunes

1. **Clave API faltante**
   ```
   Error: COHERE_API_KEY is required
   ```
   Solución: Configurar la variable de entorno en `.env`

2. **Puerto en uso**
   ```
   Error: Address already in use
   ```
   Solución: Cambiar `PORT` en `.env` o terminar el proceso existente

## 📝 TODO

- [ ] Implementar persistencia en disco
- [ ] Agregar filtros de metadata
- [ ] Implementar arquitectura leader-follower
- [ ] Crear SDK cliente de Python
- [ ] Agregar métricas y monitoring

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

**Desarrollado con ❤️ para StackAI**
