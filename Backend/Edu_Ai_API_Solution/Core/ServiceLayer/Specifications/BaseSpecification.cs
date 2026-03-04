using DomainLayer.Contracts;
using DomainLayer.Models;
using Microsoft.EntityFrameworkCore.Query;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace ServiceLayer.Specifications
{
	public abstract class BaseSpecification<Tentity,TKey>:ISpecifications<Tentity,TKey> where Tentity : BaseEntity<TKey>
	{
		protected BaseSpecification(Expression<Func<Tentity, bool>>? _Criteria)
		{
			Criteria = _Criteria;
		}
		public Expression<Func<Tentity, bool>>? Criteria { get; protected set; }
		public List<Expression<Func<Tentity, object>>> Includes { get; } = [];
		public Expression<Func<Tentity, object>> OrderBy { get; protected set; }
		public Expression<Func<Tentity, object>> OrderByDescending { get; protected set; }

        public List<Func<IQueryable<Tentity>, IIncludableQueryable<Tentity, object>>> IncludeExpressions { get; } = new();

        protected void AddInclude(Expression<Func<Tentity, object>> includeExpression)
		{
			Includes.Add(includeExpression);
		}
		protected void AddOrderBy(Expression<Func<Tentity, object>> orderByExpression)
		{
			OrderBy = orderByExpression;
		}
		protected void AddOrderByDescending(Expression<Func<Tentity, object>> orderByDescExpression)
		{
			OrderByDescending = orderByDescExpression;
		}


        

        protected void AddInclude_2(
            Func<IQueryable<Tentity>, IIncludableQueryable<Tentity, object>> includeExpression)
        {
            IncludeExpressions.Add(includeExpression);
        }

    }
}
