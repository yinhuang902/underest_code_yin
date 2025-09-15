function preprocess!(P)
    quadObj = nothing
    # provide initial value if not defined
    for i in 1:length(P.colVal)
        if isnan(P.colVal[i])
           P.colVal[i] = 0
        end
    end
    # if objective is quadratic, move to constraints


    #if( length(P.obj.qcoeffs) > 0)
        quadObj = copy(P.obj)
        @variable(P, objective_value, start=eval_g(P.obj, P.colVal))
        if P.objSense == :Min
	   if length(P.obj.qvars1) > 0
               @constraint(P, objective_value==P.obj)
	   else
	       @constraint(P, objective_value==P.obj.aff)
	   end   		   
           @objective(P, Min, objective_value)
        else
	   if length(P.obj.qvars1) > 0
               @constraint(P, objective_value==P.obj)
	   else
	       @constraint(P, objective_value==P.obj.aff)
	   end    
           @objective(P, Max, objective_value)
        end
    #end

    # provide bounds if not defined
    
    ncols =  P.numCols    
    for i in 1:ncols
        if P.colLower[i] == -Inf
            P.colLower[i] = -1e8
        end
        if P.colUpper[i] == Inf
            P.colUpper[i] = 1e8
        end
    end
    
    for i = 1:length(P.linconstr)
    	squeeze!(P.linconstr[i].terms)
    end

    # modify the constraints, so that Q is upper triangular
    for i = 1:length(P.quadconstr)
    	squeeze!(P.quadconstr[i].terms)
        con = P.quadconstr[i]
        terms = con.terms
        qvars1 = terms.qvars1
        qvars2 = terms.qvars2
        qcoeffs = terms.qcoeffs
	qVarsId_local = []
	for j in 1:length(qvars1)
	    push!(qVarsId_local, [qvars1[j].col, qvars2[j].col])	
	end
	j = 1
	while j <= length(qvars1)
                var1 = qvars1[j]
                var2 = qvars2[j]

		if var1.col > var2.col
		    qvars1[j] = var2
                    qvars2[j] = var1
		end
		index = find( x->(x == [var1.col, var2.col] || x == [var2.col, var1.col]), qVarsId_local)		

		if length(index) > 1
		   #println("index:  ", index)
		   #println("length(qvars1)    ", length(qvars1))		    		    
		   qcoeffs[j] = sum(qcoeffs[index])
		   deleteat!(index, findin(index, j))
		   deleteat!(qVarsId_local, index)
		   deleteat!(qcoeffs, index)
		   deleteat!(qvars1, index)
		   deleteat!(qvars2, index)
		end   
		j += 1
        end
    end


    # create a simple relaxed problem
    m = Model() #solver=GurobiSolver(Threads=1, LogToConsole=0))#, LogFile=string(runName,".txt")))
    # Variables
    m.numCols = P.numCols
    m.colNames = P.colNames[:]
    m.colNamesIJulia = P.colNamesIJulia[:]
    m.colLower = P.colLower[:]
    m.colUpper = P.colUpper[:]
    m.colCat = P.colCat[:]
    m.colVal = P.colVal[:]
    m.varDict = Dict{Symbol,Any}()
    for (symb,v) in P.varDict
        m.varDict[symb] = copy(v, m)
    end
    # Constraints
    m.linconstr  = map(c->copy(c, m), P.linconstr)
    # Objective
    m.obj = copy(P.obj, m)
    m.objSense = P.objSense
    # constraints
    branchVarsId = []
    qbVarsId = Dict()
    bilinearVars = []
    for i = 1:length(P.quadconstr)
            con = P.quadconstr[i]
            terms = con.terms
            qvars1 = copy(terms.qvars1, m)
            qvars2 = copy(terms.qvars2, m)
            for j in 1:length(qvars1)
		if !haskey(qbVarsId, (qvars1[j].col, qvars2[j].col)) 
                       push!(branchVarsId, qvars1[j].col)
                       push!(branchVarsId, qvars2[j].col)
                       @variable(m, bilinear)
                       push!(bilinearVars, bilinear)
		       qbVarsId[qvars1[j].col, qvars2[j].col] = bilinear.col
		       qbVarsId[qvars2[j].col, qvars1[j].col] = bilinear.col
                end
            end
    end
    branchVarsId = sort(union(branchVarsId))

    multiVariable_list = []
    multiVariable_convex = []
    multiVariable_aBB = []			 
    for i = 1:length(P.quadconstr)
            con = P.quadconstr[i]
            terms = con.terms
            qvars1 = terms.qvars1
            qvars2 = terms.qvars2
	    qcoeffs = terms.qcoeffs
	    aff = terms.aff
	    mvs = MultiVariable[]

	    remain=collect(1:length(qvars1))	    
	    while length(remain) != 0
	    	mv = MultiVariable()
		mvTerms = mv.terms
                push!(mvs, mv)
		
		seed = remain[1]
		var1 = qvars1[seed]
                var2 = qvars2[seed]
                coeff = qcoeffs[seed]
                push!(mvTerms.qvars1, var1)
                push!(mvTerms.qvars2, var2)
                push!(mvTerms.qcoeffs, coeff)
                push!(mv.qVarsId, var1.col)
                push!(mv.qVarsId, var2.col)
                mv.qVarsId = sort(union(mv.qVarsId))				
		deleteat!(remain, 1)
		while length(remain) != 0
		    added = []
		    for k in 1:length(remain)
		    	j = remain[k]
		    	var1 = qvars1[j]
                	var2 = qvars2[j]
                	coeff = qcoeffs[j]
			index = findin(mv.qVarsId,var1.col)
			if index == []
			    index = findin(mv.qVarsId,var2.col)
			end
			if length(index) != 0
                	    push!(mvTerms.qvars1, var1)
                	    push!(mvTerms.qvars2, var2)
                	    push!(mvTerms.qcoeffs, coeff)
                	    push!(mv.qVarsId, var1.col)
                	    push!(mv.qVarsId, var2.col)
                	    mv.qVarsId = sort(union(mv.qVarsId))
			    push!(added, k)
			end
		    end  
		    deleteat!(remain, added)
		    if length(added) == 0
		        break
		    end
		end		  	  
	    end

	    affcopy = copy(aff)
	    nonlinearIndexs = []
	    for j in 1:length(aff.vars)
	    	var = aff.vars[j]
		coeff = aff.coeffs[j]
		index = find( x->( length(findin(x.qVarsId,var.col))!= 0), mvs)	    	
		if length(index) != 0
	    	    mv = mvs[index[1]]
		    push!(mv.terms.aff.vars, var)
		    push!(mv.terms.aff.coeffs, coeff)
		    push!(nonlinearIndexs, j)	
		end    
	    end
            deleteat!(affcopy.vars, nonlinearIndexs)
            deleteat!(affcopy.coeffs, nonlinearIndexs)
	    
	    for j in 1:length(mvs)
	    	 mv = mvs[j]
		 mvTerms = mv.terms
		 C = length(mv.qVarsId)
		 Q = zeros(Float64, C, C)

		 for k in 1:length(mv.qVarsId)
		     push!(mv.qVars, Variable(m, mv.qVarsId[k]))
		 end		 

		 for k in 1:length(mvTerms.qvars1)		 
		     var1 = mvTerms.qvars1[k]		 
		     var2 = mvTerms.qvars2[k]	  
		     coeff = mvTerms.qcoeffs[k]		     
		     I = findin(mv.qVarsId,var1.col)[1]
		     J = findin(mv.qVarsId,var2.col)[1]	  
		     Q[I, J] = coeff
		     if haskey(qbVarsId, (var1.col, var2.col))
		     	 bid = qbVarsId[var1.col, var2.col]
		     end	 		 
		     push!(mv.bilinearVars, Variable(m, bid))		     
		 end	
		 mv.Q = copy(Q)
		 Q = (Q+Q')/2
		 val, vec = eig(Q)
		 if sum(val.>=0) == C
		    mv.pd = 1
		    push!(multiVariable_convex, mv)
		 end
                 if sum(val.<=0) == C
                    mv.pd = -1
		    push!(multiVariable_convex, mv)
		 end
		 if sum(val.==0) == C
                    mv.pd = 0
		 end   		    
		 if mv.pd == 0 
		    alpha = Array(Float64, C)
		    lamda_min = minimum(val)
		    added = true

		    if sum(diag(Q).==0) == C
		        if length(mv.qVarsId) < 3
			   added = false
			end
		    end
		    for k in 1:C
		    	alpha[k] = max(0,  min(-lamda_min, sum(abs(Q[k,:])) - 2*Q[k,k])) 
			varId =  mv.qVarsId[k]
                        if !haskey(qbVarsId, (varId, varId))
			    added = false
			    break
			end			
		    end
		    #println("alpha  ", alpha) 
		    #println("added  ", added)
		    if added 
		        mv.alpha = alpha
		        push!(multiVariable_aBB, mv)
		    end
		 end		 
	    end
	    push!(multiVariable_list, MultiVariableCon(mvs, copy(affcopy)))
    end


    # create RLT
    EqVconstr = LinearConstraint[]      
    

    for i = 1:length(P.linconstr)
        con = P.linconstr[i]
        if con.lb == con.ub
	    constant = con.lb
            for varId in branchVarsId
                newcon = copy(con, m)
                aff = newcon.terms
		accept = true
                for k in 1:length(aff.vars)
                    varIdinEq = aff.vars[k].col
                    if haskey(qbVarsId, (varId, varIdinEq))
		       aff.vars[k] = Variable(m,qbVarsId[varId, varIdinEq])
		    else
		        accept = false
                        break
		    end 
                end
		if accept
		    if constant != 0
		       push!(aff.vars, Variable(m, varId))		    
		       push!(aff.coeffs, -constant)
		       newcon.lb = 0
		       newcon.ub = 0
		    end   
                    push!(EqVconstr, newcon)
		end
            end	    
        end
    end


    pr = PreprocessResult()
    pr.branchVarsId = branchVarsId
    pr.qbVarsId = qbVarsId
    pr.EqVconstr = EqVconstr
    pr.multiVariable_list = multiVariable_list
    pr.multiVariable_convex = multiVariable_convex
    pr.multiVariable_aBB = multiVariable_aBB
    return pr
end

function findnode(c, nlinconstrs)
    nodeid = 1	 
    for i in 1:length(nlinconstrs)
	ncon = nlinconstrs[i]
	if c <= ncon  
	    nodeid = i
	    break
	end
    end
    return nodeid
end



function squeeze!(aff::AffExpr)
     redund = true
     while redund
         redund = false
         for k in 1:length(aff.vars)
             x = aff.vars[k]
             equal_index = find(aff.vars .== x)
             if length(equal_index) > 1
                 aff.coeffs[k] = sum(aff.coeffs[equal_index])
                 deleteat!(aff.vars, equal_index[2:end])
                 deleteat!(aff.coeffs, equal_index[2:end])
                 redund = true
                 break
             end
         end
     end
end

function squeeze!(quad::QuadExpr)
    qvars1 = quad.qvars1
    qvars2 = quad.qvars2
    qcoeffs = quad.qcoeffs
    aff = quad.aff
    redund = true
    while redund
         redund = false
         for k in 1:length(qvars1)
             x = qvars1[k]
             y = qvars2[k]

             equal_index = []
             coeff = 0
             for j in 1:length(qvars1)
                 if (qvars1[j] == x && qvars2[j] == y)|| (qvars1[j] == y && qvars2[j] == x)
                     push!(equal_index, j)
                     coeff += qcoeffs[j]
                 end
             end

             if length(equal_index) > 1
                 qcoeffs[k] = coeff
                 deleteat!(qvars1, equal_index[2:end])
                 deleteat!(qvars2, equal_index[2:end])
                 deleteat!(qcoeffs, equal_index[2:end])
                 redund = true
                 break
             end
         end
     end
     squeeze!(aff)
end