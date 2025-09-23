# StackAI Vector Database

Una API REST para indexar y consultar documentos en una base de datos vectorial, desarrollada como parte del proceso de entrevista técnica de StackAI.

## 🚀 Características

- **API REST completa** para operaciones CRUD en bibliotecas, documentos y chunks
- **Búsqueda vectorial k-NN** con múltiples algoritmos de indexación
- **Arquitectura limpia** siguiendo principios DDD y SOLID
- **Logging estructurado** con request tracking y múltiples formatters
- **Request middleware** con IDs únicos y métricas de timing
- **Tipado estático** completo con MyPy
- **Containerización** con Docker
- **Herramientas de desarrollo** integradas (Black, Ruff, Pre-commit)
- **Suite de tests completa** con cobertura de componentes principales

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

# Ejecutar tests específicos
pytest tests/test_health.py -v
pytest tests/test_config.py -v
pytest tests/test_logging.py -v

# Con cobertura
pytest tests/ --cov=app --cov-report=html

# Ejecutar solo tests unitarios
pytest tests/ -m "not integration" -v
```

### Tests Incluidos

- **test_health.py**: Tests para endpoints de health check y middleware
- **test_config.py**: Tests para configuración de Pydantic Settings
- **test_logging.py**: Tests para sistema de logging estructurado
- **test_main.py**: Tests para aplicación principal y middleware
- **test_schemas.py**: Tests para validación de esquemas Pydantic

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

### Variables Principales

- `COHERE_API_KEY`: Clave API de Cohere (requerida)
- `DEFAULT_INDEX_TYPE`: Tipo de índice por defecto (linear, kdtree, ivf)
- `MAX_CHUNKS_PER_LIBRARY`: Máximo número de chunks por biblioteca
- `LOG_LEVEL`: Nivel de logging (DEBUG, INFO, WARNING, ERROR)

### Logging Configuración

- `LOG_FORMAT_GENERAL`: Formato para logs generales
- `LOG_FORMAT_REQUEST`: Formato para logs de requests con campos estructurados
- Logging estructurado con request IDs únicos
- Múltiples handlers y formatters configurados via dictConfig

Ver `env.example` para todas las opciones disponibles.

## 🔍 Algoritmos de Indexación

### Implementados

#### 1. LinearScanIndex (Baseline)
- **Complejidad Temporal**: 
  - Build: O(N) - Almacena vectores en memoria
  - Query: O(N × D) - Escaneo exhaustivo
  - Add/Remove: O(1) - Operaciones directas en lista
- **Complejidad Espacial**: O(N × D)
- **Características**:
  - Resultados exactos (sin aproximación)
  - Sin preprocesamiento requerido
  - Excelente para datasets pequeños (<1K vectores)
  - Baseline confiable para comparación

#### 2. KDTreeIndex (Eficiente en Bajas Dimensiones)
- **Complejidad Temporal**:
  - Build: O(N log N) - Particionado recursivo con búsqueda de mediana
  - Query: O(log N) promedio, O(N) peor caso
  - Add/Remove: O(log N) - Puede requerir rebalanceo
- **Complejidad Espacial**: O(N) - Estructura de árbol
- **Características**:
  - Excelente para dimensiones bajas (D ≤ 20)
  - Performance se degrada en altas dimensiones (maldición de dimensionalidad)
  - Resultados exactos
  - Estructura de memoria eficiente

#### 3. IVFIndex (Inverted File - Escalable)
- **Complejidad Temporal**:
  - Build: O(N × C × I) - N vectores, C clusters, I iteraciones k-means
  - Query: O(P × M + k) - P sondeos, M vectores promedio por cluster
  - Add: O(C) - Encontrar cluster más cercano
  - Remove: O(M) - Buscar en lista invertida
- **Complejidad Espacial**: O(N + C × D) - Vectores + centroides
- **Características**:
  - Excelente escalabilidad para datasets grandes (>10K vectores)
  - Resultados aproximados (ajustable vía parámetro nprobe)
  - Buen rendimiento en altas dimensiones
  - Actualizaciones incrementales eficientes

### Selección Automática de Algoritmo

El sistema selecciona automáticamente el algoritmo óptimo basado en:

- **Datasets pequeños** (<1K vectores): LinearScan
- **Dimensiones bajas** (D ≤ 20, <50K vectores): KDTree  
- **Datasets grandes** (≥10K vectores) o **altas dimensiones** (D > 50): IVF
- **Prioridad de precisión**: KDTree para dim bajas, LinearScan para dim altas
- **Prioridad de velocidad**: IVF para la mayoría de casos

### Justificación de Selección

**¿Por qué estos 3 algoritmos?**

1. **LinearScan**: Baseline esencial que garantiza resultados exactos y sirve como referencia de correctitud
2. **KDTree**: Algoritmo clásico que demuestra técnicas de particionado espacial, excelente para casos de uso específicos
3. **IVF**: Algoritmo moderno usado en sistemas de producción (similar a FAISS), escalable y práctico

Esta combinación cubre el espectro completo: exactitud vs velocidad, datasets pequeños vs grandes, y dimensiones bajas vs altas.

### Consideraciones de Concurrencia

- **Single-writer principle**: Operaciones de escritura protegidas con locks
- **Read/Write locks**: Múltiples lectores concurrentes permitidos
- **Thread-safe wrapper**: Envoltorio automático para operaciones concurrentes
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

### Próximas Funcionalidades
- [ ] Crear servicios de aplicación para búsqueda k-NN
- [ ] Implementar endpoints REST para búsqueda vectorial
- [ ] Integrar indexación automática en LibraryService

### Mejoras Futuras (Extras)
- [ ] Implementar persistencia en disco
- [ ] Agregar filtros de metadata
- [ ] Implementar arquitectura leader-follower
- [ ] Crear SDK cliente de Python
- [ ] Agregar métricas y monitoring

### ✅ Completado
- [x] Estructura base del proyecto con arquitectura DDD
- [x] Configuración de herramientas de desarrollo (Black, Ruff, MyPy)
- [x] Sistema de logging estructurado con múltiples formatters
- [x] Request middleware con tracking de IDs únicos
- [x] Suite de tests completa para componentes base
- [x] Containerización con Docker
- [x] Health check endpoint funcional
- [x] **Modelos de dominio completos** (Library, Document, Chunk)
- [x] **Algoritmos de indexación implementados** (Linear, KD-Tree, IVF)
- [x] **Cliente Cohere para embeddings** con fallback a FakeClient
- [x] **Endpoints REST CRUD completos** para todas las entidades
- [x] **Sistema de embeddings integrado** con dependency injection
- [x] **Thread-safety y concurrencia** en algoritmos de indexación

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
