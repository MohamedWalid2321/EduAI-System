using DomainLayer.Models;
using Microsoft.EntityFrameworkCore.Query;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Contracts
{
	public interface ISpecifications<Tentity,TKey> where Tentity : BaseEntity<TKey>
	{
		Expression<Func<Tentity, bool>>? Criteria { get; }
        List<Expression<Func<Tentity, object>>> Includes { get; }

        List<Func<IQueryable<Tentity>, IIncludableQueryable<Tentity, object>>> IncludeExpressions { get; }



        Expression<Func<Tentity, object>> OrderBy { get; }
		Expression<Func<Tentity, object>> OrderByDescending { get; }

	}
}
