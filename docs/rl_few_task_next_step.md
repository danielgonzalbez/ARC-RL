# Siguiente paso: diagnóstico RL de 16 familias

## Decisión

Las pruebas de una y cuatro familias ya demuestran que el entorno, la reward,
el optimizador y la sincronización con vLLM pueden aprender. El siguiente paso
controlado es entrenar sobre 16 familias del lado RL de la partición 60/40 y
evaluar transformaciones no vistas de esas mismas 16 familias. Saltar de cuatro
directamente a 144 familias hizo más difícil atribuir los cambios observados.

El manifiesto canónico es
`data_splits/qwen35_rl_diagnostic_16tasks_aug_holdout.json`. Conserva las cuatro
familias anteriores y añade doce familias mediante cobertura determinista de
la longitud serializada. El rango total es 211--7.972 tokens.

Por familia se usan:

- train: original, rotaciones 90/180/270 y recoloreados con seeds 3408/3409;
- eval: recoloreados independientes con seeds 3410/3411.

Esto produce 96 ejemplos de train y 32 de eval. El generador del manifiesto
comprueba que no exista ninguna colisión de contenido serializado entre ambos
lados. El loader vuelve a comprobarlo antes de crear el modelo.

## Experimento principal

El launcher `run_qwen35_rl_diagnostic_16tasks.sh` parte del checkpoint SFT
disponible, usa RL puro (`supervised_anchor_weight=0`) y mantiene vLLM tanto
para rollouts como para evaluación generativa. La mezcla inicial es la variante
suave sugerida por el ensayo de 144 familias:

Las exportaciones reproducibles del motor se escriben por defecto bajo
`/tmp/arc_vllm_work`, no en el directorio durable de checkpoints. Si una
ejecución anterior dejó una exportación con el mismo nombre, el runner elige
automáticamente un sufijo `_restartNN` y conserva el snapshot anterior.

```text
pair_correctness_weight = 0.75
structured_f1_weight    = 0.25
```

Ejecutar:

```bash
bash run_qwen35_rl_diagnostic_16tasks.sh
```

La comparación controlada debe partir del mismo checkpoint y cambiar solamente
el término F1:

```bash
RUN_NAME=qwen35_rl_diagnostic_16tasks_control_v1 \
PAIR_WEIGHT=0.75 STRUCTURED_F1_WEIGHT=0 \
bash run_qwen35_rl_diagnostic_16tasks.sh
```

No se debe comparar una rama reanudada desde otra con el control.

## Criterio para avanzar

Una configuración avanza si mejora al menos `+0.01` el F1 greedy del holdout
frente a su evaluación del paso 0, no reduce exactos ni EOS, y la dirección es
corroborada por F1 muestreado o `pass@4`. Entre configuraciones se elige la que
mantenga mejor el resultado, no un pico aislado.

Si ninguna rama pasa, el problema se investiga aquí variando LR, número de
rollouts o mezcla de rewards. Si una pasa, se repite con 32 familias y el mismo
protocolo de vistas disjuntas. Solo después se abre de nuevo el conjunto fijo
de 150 tareas para una evaluación final con vLLM.

## Estado del ensayo de 144 familias

El log `qwen35_arc_full8192_rl_full_aug_top8_mix_v9.jsonl` no contiene un evento
final: terminó tras el update 55 de 72. Su mejor evaluación registrada fue el
paso 54, con 4/150 exactos y F1 micro 0.703026, frente al baseline 3/150 y
0.700662. Es una señal positiva pequeña, pero el checkpoint citado por el log
no está presente en esta copia y la corrida no puede considerarse terminada ni
reanudable.
