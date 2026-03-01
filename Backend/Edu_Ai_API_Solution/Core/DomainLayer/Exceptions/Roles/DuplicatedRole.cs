using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Roles
{
	public sealed class DuplicatedRole(): ConflictException("Another role with the same name is already exists")
	{

	}
}
