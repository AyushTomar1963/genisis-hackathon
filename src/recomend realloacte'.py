  
    def perform_reallocation(self, congestion_probs, current_loads):
        """
        Main reallocation algorithm - Greedy approach
        """
        loads = current_loads.copy()  # Don't modify original
        reallocations = []
        processed_nodes = set()  # Avoid infinite loops
        
        for iteration in range(self.max_iterations):
            # Step 1: Detect congested nodes
            congested_nodes = self.detect_congested_nodes(congestion_probs, loads)
            
            if not congested_nodes:
                print(f"✅ Reallocation complete in {iteration} iterations")
                break
            
            reallocation_made = False
            
            # Step 2: For each congested node, find targets
            for source in congested_nodes:
                if source['node_id'] in processed_nodes:
                    continue
                    
                # Find available target nodes
                target_nodes = self.find_target_nodes(
                    congestion_probs, loads, {source['node_id']}
                )
                
                if not target_nodes:
                    continue
                
                # Step 3: Transfer to best target
                best_target = target_nodes[0]  # Lowest congestion prob
                transfer_amount = self.calculate_transfer_amount(source, best_target)
                
                if transfer_amount > 0.01:  # Minimum transfer threshold
                    # Execute transfer
                    loads[source['node_id']] -= transfer_amount
                    loads[best_target['node_id']] += transfer_amount
                    
                    reallocations.append({
                        'from_node': source['node_id'],
                        'to_node': best_target['node_id'],
                        'amount': transfer_amount,
                        'iteration': iteration
                    })
                    
                    reallocation_made = True
                    print(f"🔄 Transfer {transfer_amount:.3f} from Node-{source['node_id']} to Node-{best_target['node_id']}")
                
                processed_nodes.add(source['node_id'])
            
            if not reallocation_made:
                print(f"⚠️ No more transfers possible at iteration {iteration}")
                break
        
        return loads, reallocations