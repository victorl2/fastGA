#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <functional>
#include <iterator>
#include <random>
#include "trade_model.hpp"


namespace GA
{	
	std::random_device rd2;    
	std::mt19937 gen(rd2());

	enum CROSSOVER { ONE_POINT, TWO_POINT, UNIFORM };
	enum SELECTION { ROULETTE_WHEEL, RANK, FITNESS_UNIFORM };
	enum DELETION_POLICY{ WORSE_FITNESS, FUDS };

	double random(){ return ((double) rand() / (RAND_MAX)); }

	class Individual{
		public:
			std::string cromossome;
			double fitness;
			Individual(std::string new_cromossome): cromossome(new_cromossome){}
	};

	bool comparison(Individual i,Individual j) { return (i.fitness < j.fitness ); }

	class GeneticAlgorithm {
		public:
			SELECTION selection_type = FITNESS_UNIFORM;
			CROSSOVER crossover_type = UNIFORM;
			DELETION_POLICY deletion_type = WORSE_FITNESS;

			bool verbose = false;
			double fuds_base_distance = 0.03;
			double percentage_removal = 0.15;
			double mutation_rate = 0.1; //the probability of a individual to be mutated - [0,1]
			unsigned int mutation_bits = 1; //the average number of bits that will be mutate in a individual when mutation occurs
			unsigned int population_size = 100; // the desired population size
			unsigned int seed = 95742; // random seed used to generate random values

			unsigned cromossome_size;
			unsigned int generations = 10; // total amount of generations to execute the algorithm

			std::vector<std::string> initial_population; // initial Individuals that can be provided to training
			double (*fitness_function)(std::string); // pointer to a fitness function receiving a cromossome string
			bool (*is_cromossome_valid)(std::string); //function to check if a cromossome is valid
			void (*report)(unsigned int, std::vector<Individual>); //function to report results of each generation

			GeneticAlgorithm(){}

			std::vector<Individual> run(){
				srand(seed);
				
				std::cout << "Generating initial poulation" << std::endl;
				std::vector<Individual> current_population = generate_initial_population();
				std::cout << "calculating fitness" << std::endl;
				evaluate_fitness(current_population);
				std::cout << "sorting vector" << std::endl;
				std::sort(current_population.begin(), current_population.end(), comparison);

				for(unsigned int current_gen=0; current_gen < generations; current_gen++){
					
					
					std::cout << ">>Trimming population" << std::endl;
					std::vector<Individual> trimmed_population = trim_population(current_population);
					std::cout << ">>Generating children" << std::endl;
					
					std::vector<Individual> new_children = generate_new_individuals(population_size - trimmed_population.size() ,
								current_population);
					
					std::cout << ">>calculating fitness" << std::endl;
					evaluate_fitness(new_children);
					
					trimmed_population.insert(trimmed_population.end(), new_children.begin(), new_children.end());
					current_population = trimmed_population;
					
					std::cout << ">>sorting vector" << std::endl << std::endl;

					std::sort(current_population.begin(), current_population.end(), comparison);
					report(current_gen, current_population);
				}

				return current_population;
			}


			std::vector<Individual> trim_population(std::vector<Individual>& current_population){
				std::vector<Individual> new_population;
				if(deletion_type == WORSE_FITNESS){
					int amount_to_remove = (int) current_population.size() * percentage_removal;
					new_population.reserve(amount_to_remove);
					new_population.insert(new_population.end(), current_population.begin() + amount_to_remove, current_population.end());
					return new_population;
				}

				if(deletion_type == FUDS){
					double distance = fuds_base_distance;
					
					std::vector<Individual> individuals;
					
					do {
						individuals.clear();
						distance *= 1.1;
						
						for(int i=current_population.size()-1; i > 0; i--){
							if(std::abs(current_population[i].fitness - current_population[i -1].fitness) > distance){
								individuals.push_back(current_population[i]); 
							}
						}
					} while(individuals.size() >= (current_population.size() * (1 - percentage_removal) ));
					
					std::cout << "Distance used: " << distance << std::endl;
					std::cout << "Individuals removed: " << current_population.size() - individuals.size() << std::endl;
					std::cout << "Desired removal: " << current_population.size() *  percentage_removal << std::endl;
					return individuals;
				}

				return current_population;
			}

			std::vector<Individual> generate_new_individuals(int amount_to_generate,std::vector<Individual> current_population){
				std::vector<Individual> new_individuals;

				while(new_individuals.size() < amount_to_generate){
					std::vector<Individual> parents = selection(current_population);
					Individual new_individual = cross_over(parents[0], parents[1]);
					if(random() <= mutation_rate)
						new_individual = mutate(new_individual);
					new_individuals.push_back(new_individual);
				}
				return new_individuals;
			}

		private:
			std::vector<Individual> generate_initial_population(){
				std::vector<Individual> population;
				for(int i=0; i<initial_population.size(); i++){
					population.push_back(Individual(initial_population[i]));
				}

				if(population.size() >= population_size){
					return population;
				}
				for(int i=0; i< population_size - population.size(); i++){
					population.push_back(Individual(generate_valid_cromossome()));
				}
				return population;
			}

			std::string generate_valid_cromossome(){
				std::string cromossome;
				do{
					cromossome = generate_cromossome();
				}while(!(*is_cromossome_valid)(cromossome));
				return cromossome;	
			}

			Individual cross_over(Individual& father, Individual& mother){
				switch(crossover_type){
					case UNIFORM:
						return uniform_crossover(father, mother);
					case ONE_POINT:
						return one_point_crossover(father, mother);
					case TWO_POINT:
						return two_point_crossover(father, mother);
					default:
						return one_point_crossover(father, mother);
				}
			}

			Individual uniform_crossover(Individual& father, Individual& mother){
				std::string new_cromossome;
				for(int i=0; i < father.cromossome.size(); i++){
					new_cromossome += random() < 0.5 ? father.cromossome[i] : mother.cromossome[i];
				}
				return new_cromossome;
			}

			Individual one_point_crossover(Individual& father, Individual& mother){
				int cut_position = random() * father.cromossome.length() -2;
				return Individual(father.cromossome.substr(0,cut_position) + 
					mother.cromossome.substr(cut_position + 1, mother.cromossome.length() - cut_position));
			}

			Individual two_point_crossover(Individual& father, Individual& mother){
				int cut_position1 = random() * father.cromossome.length() -2;
				int cut_position2 = cut_position1 + (random() * (father.cromossome.length() - cut_position1 - 1));

				return Individual(father.cromossome.substr(0, cut_position1) + 
					mother.cromossome.substr(cut_position1 + 1, cut_position2 - (cut_position1 + 1) ) +  
					father.cromossome.substr(cut_position2 + 1, father.cromossome.length() - (cut_position2 + 1)) );
			}

			Individual mutate(Individual& ind){
				std::string new_cromossome;
				double mutation_probability = (double)mutation_rate/ind.cromossome.size();
				for (std::string::size_type i = 0; i < ind.cromossome.size(); i++) {
					if(random() <= mutation_probability ){
						new_cromossome += ind.cromossome[i] == '0' ? '1': '0'; 
					}else{
						new_cromossome += ind.cromossome[i];
					}
				}
				return Individual(new_cromossome);
			}

			//returns two individuals to be used as parents to generate a new solution
			std::vector<Individual> selection(std::vector<Individual>& population){
				if(selection_type == FITNESS_UNIFORM){
					double target_fitness = population[0].fitness + random() * population[population.size()-1].fitness;
					int pos_father = find_position_closest_value(target_fitness, population);
					int pos_mother = 0;
					do{
						target_fitness = population[0].fitness + random() * population[population.size()-1].fitness;
						pos_mother = find_position_closest_value(target_fitness, population);
					}while(pos_mother == pos_father);

					return std::vector<Individual>{population[pos_father], population[pos_mother]};
				}
				return population;
			}

			void evaluate_fitness(std::vector<Individual>& population){
				for(int i = 0; i < population.size(); i++){
					if(!(*is_cromossome_valid)(population[i].cromossome)){
						population.erase(population.begin() + i);
					}
				}
				
				#pragma omp parallel for
				for(int i = 0; i < population.size(); i++){
					population[i].fitness = (*fitness_function)(population[i].cromossome);	
				}				
			}

			int find_position_closest_value(double target_fitness_value, std::vector<Individual>& population){
				if(target_fitness_value < population[0].fitness) {
					return 0;
				}
				if(target_fitness_value > population[population.size()-1].fitness) {
					return population.size()-1;
				}

				double best_distance = std::abs(target_fitness_value - population[0].fitness);
				double last_distance = best_distance;
				int pos = 0;
				for(int i=1; i<population.size(); i++){
					double current_distance = std::abs(target_fitness_value - population[i].fitness);
					if(current_distance > last_distance) return i-1;
					if(current_distance < best_distance){
						best_distance = current_distance;
						pos = i;
					}
				}
				return pos;
			
			}
			
	};

	
}