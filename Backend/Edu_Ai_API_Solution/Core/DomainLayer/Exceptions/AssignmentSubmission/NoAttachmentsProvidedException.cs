using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.AssignmentSubmission
{
    public class NoAttachmentsProvidedException():BadRequestException("No attachments provided for the assignment submission.")
    {
    }
}
