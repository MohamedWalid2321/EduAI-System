using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions
{
    public sealed class AssignmentAttachmentNotFoundException(Guid id): NotFoundException($"Assignment Attachment with this Id : {id} is Not Found ")
    {
    }
}
