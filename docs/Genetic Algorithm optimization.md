
## Inicialización de las variables de decisión

- En cada iteración de la simulación, excepto en la inicial se pueden reutilizar los mejores candidatos de la solución en la iteración anterior (no quedarse solo con el individuo más óptimo, si no de alguna manera extraer todo el espacio de soluciones en la última iteración). 
  Si los muestreos cambian entre iteraciones (ver [Variable sample rate](Resolution%20strategy.md#Variable%20sample%20rate)), es cuestión de determinar la nueva distribución, y después re-muestrear la solución anterior a la nueva distribución.
- En cuanto a la primera iteración, lo suyo sería implementar una estrategia que almacene soluciones en una base de datos y haya una estrategia para identificar soluciones previas para usar como punto de partida. 
  Como antes, no solo guardando la solución *óptima*, si no todos los candidatos finales para tener más riqueza.
  No debería de haber problemas por mezclar en la misma base de datos soluciones con distintos parámetros del modelo, puesto que en la selección esto se tendrá en cuenta.

### Selección de soluciones previas

- Costes parecidos
- Perfil de variables de entorno parecido (I, Tamb, Tc,in):

e.g.: Calcular diferencia entre la I actual (Iref) y la de los candidatos:

$$\left|I_{ref} - I_{candidates}\right| = \left| I_{ref} - \begin{bmatrix} I_{A,0} & I_{A,1} & ... & I_{A,N_p} \\ & & \vdots & \\ I_{Z,0} & I_{Z,1} & ... & I_{Z,N_p} \end{bmatrix} \right|$$

Quedarse con candidatos con diferencias menores

## PyGAD: Genetic Algorithm in Python

Basically copied some parts of the [project's](https://github.com/ahmedfgad/GeneticAlgorithmPython) documentation and README.

Using the `pygad` module, instances of the genetic algorithm can be created, run, saved, and loaded. Single-objective and multi-objective optimization problems can be solved.

- Supports optimizing both single-objective and multi-objective problems.
- Supports different types of crossover, mutation, and parent selection
- Support different variable types for each decision variable [Data Type for each Individual Gene without Precision](#Data%20Type%20for%20each%20Individual%20Gene%20without%20Precision%20([source](https%20//pygad.readthedocs.io/en/latest/pygad_more.html%20data-type-for-each-individual-gene-without-precision)).
- Support for specifying the solution space for each decision variable [Gene solution space](#Gene%20solution%20space%20([source](https%20//pygad.readthedocs.io/en/latest/pygad.html%20init)).
- Support for setting limits on the decision variables [Decision variable limits](#Decision%20variable%20limits).




### Data Type for each Individual Gene without Precision ([source](https://pygad.readthedocs.io/en/latest/pygad_more.html#data-type-for-each-individual-gene-without-precision))

In [PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0), the `gene_type` parameter allows customizing the gene type for each individual gene. This is by using a `list`/`tuple`/`numpy.ndarray` with number of elements equal to the number of genes. For each element, a type is specified for the corresponding gene.

This is an example for a 5-gene problem where different types are assigned to the genes.

```
gene_type=[int, float, numpy.float16, numpy.int8, float]
```


### Gene solution space ([source](https://pygad.readthedocs.io/en/latest/pygad.html#init))

The parameter `gene_space=None` is used to specify the possible values for each gene in case the user wants to restrict the gene values. It is useful if the gene space is restricted to a certain range or to discrete values. It accepts a `list`, `range`, or `numpy.ndarray`. When all genes have the same global space, specify their values as a `list`/`tuple`/`range`/`numpy.ndarray`. For example, `gene_space = [0.3, 5.2, -4, 8]` restricts the gene values to the 4 specified values. If each gene has its own space, then the `gene_space` parameter can be nested like `[[0.4, -5], [0.5, -3.2, 8.2, -9], ...]` where the first sublist determines the values for the first gene, the second sublist for the second gene, and so on. If the nested list/tuple has a `None` value, then the gene’s initial value is selected randomly from the range specified by the 2 parameters `init_range_low` and `init_range_high` and its mutation value is selected randomly from the range specified by the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. `gene_space` is added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0). Check the [Release History of PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) section of the documentation for more details. In [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0), NumPy arrays can be assigned to the `gene_space` parameter. In [PyGAD 2.11.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0), the `gene_space` parameter itself or any of its elements can be assigned to a dictionary to specify the lower and upper limits of the genes. For example, `{'low': 2, 'high': 4}` means the minimum and maximum values are 2 and 4, respectively. In [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0), a new key called `"step"` is supported to specify the step of moving from the start to the end of the range specified by the 2 existing keys `"low"` and `"high"`.

### Decision variable limits

There seem to be multiple ways of setting the solution space limits:

- `init_range_low` and `init_range_high` initialization arguments for the gene’s initial value.
- `random_mutation_min_val` and `random_mutation_max_val` for the gene's subsequent values consequence of mutations (not sure why it needs to be different from the initial ones?)
- **Preferred option** Use a dict `{'low': 2, 'high': 4}`, optionally with the key `step` to restrict the number of possible values.

**UPDATE: Está explicado bastante regular, `random_mutation...` se refiere a la variación, no al valor absoluto, por eso tienen valores distintos. Entonces:**
- `init_range_*` hace referencia  a los límites absolutos de una variable
- `random_mutation_*` hace referencia a la máxima variación durante la mutación de una variable respecto a su valor previo a la mutación

Docs ([source](https://pygad.readthedocs.io/en/latest/pygad.html#init)):

- `init_range_low=-4`: The lower value of the random range from which the gene values in the initial population are selected. `init_range_low` defaults to `-4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.
    
- `init_range_high=4`: The upper value of the random range from which the gene values in the initial population are selected. `init_range_high` defaults to `+4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.

- `random_mutation_min_val=-1.0`: For `random` mutation, the `random_mutation_min_val` parameter specifies the start value of the range from which a random value is selected to be added to the gene. It defaults to `-1`. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`. 
    
- `random_mutation_max_val=1.0`: For `random` mutation, the `random_mutation_max_val` parameter specifies the end value of the range from which a random value is selected to be added to the gene. It defaults to `+1`. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`.

#### Dynamic decision variable limits

Do we really care about this? Since the algorithm is providing the solution space **before** evaluating the model along the prediction horizon, there is no way to give it feedback about the values. 

The way the model itself is constructed, when a decision variable is above the operational limit, it automatically sets it to its upper limit, and if below... to the minimum value? (need to check if its to the minimum or zero), so in principle there should be no errors. And in the end what will determine the final value of the optimization will be the fitness function.

## Initial values for the solution space

- `initial_population`: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to `None` which means no initial population is specified by the user. In this case, [PyGAD](https://pypi.org/project/pygad) creates an initial population using the `sol_per_pop` (número de individuos, en MATLAB por defecto son 100/150? (comprobar)) and `num_genes` (nuestro $L_c$, número de variables de decisión) parameters. An exception is raised if the `initial_population` is `None` while any of the 2 parameters (`sol_per_pop` or `num_genes`) is also `None`. Introduced in [PyGAD 2.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-0-0) and higher.
    
- `sol_per_pop`: Number of solutions (i.e. chromosomes) within the population. This parameter has no action if `initial_population` parameter exists.
    
- `num_genes`: Number of genes in the solution/chromosome. This parameter is not needed if the user feeds the initial population to the `initial_population` parameter.

### Partial initialization

In MATLAB, one can provide a matrix of initial solutions, and the rest is filled by the default random initialization strategy. Is that supported here?

Comprobar si está soportado, si no quizás sea fácil de implementar buscando cómo lo hace por defecto, y llamarlo para rellenar el tamaño que falte. Hacer un PR añadiendo esto.

### Life Cycle of PyGAD

![center | 300](attachments/Pasted%20image%2020240510103455.png)

The next figure lists the different stages in the lifecycle of an instance of the `pygad.GA` class. Note that PyGAD stops when either all generations are completed or when the function passed to the `on_generation` parameter returns the string `stop`.


## Algorithm options

Aquí explicar lo entendido de cómo funciona cada parámetro, poner los valores por defecto de la librería e indicar los valores escogidos y por qué.

- num_generations: None  
- num_parents_mating: None  
- fitness_func: None  
- fitness_batch_size: -4  
- initial_population: 4  
- sol_per_pop: <class 'float'>  
- num_genes: sss  
- init_range_low: -1  
- init_range_high: 1  
- gene_type: 3  
- parent_selection_type: single_point  
- keep_parents: None  
- keep_elitism: random  
- K_tournament: None  
- crossover_type: False  
- crossover_probability: default  
- mutation_type: None  
- mutation_probability: -1.0  
- mutation_by_replacement: 1.0  
- mutation_percent_genes: None  
- mutation_num_genes: True  
- gene_space: None  
- allow_duplicate_genes: None  
- on_start: None  
- on_fitness: None  
- on_parents: None  
- on_crossover: 0.0  
- on_mutation: False  
- on_generation: False  
- on_stop: False  
- delay_after_gen: None  
- save_best_solutions: None  
- save_solutions: None


### Possible contributions to PyGAD

- [ ] Add support for partial initialization
- [ ] Improve documentation regarding the decision variable limits and mutation variation limits
- [ ] Why are the default limits set to -4? They should be set to an arbitrary large number or force the user to specify it, otherwise it might lead to unintended errors
- [ ] Passing methods to the PyGAD instance
- [ ] Documentation should be segmented more, there is too much content on the same page